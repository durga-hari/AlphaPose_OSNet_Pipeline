# -*- coding: utf-8 -*-
"""
alphapose_wrapper.py — Robust AlphaPose SPPE loader/decoder.
- Adapts to builder APIs: build_sppe / get_pose_net / get_model
- Loads HRNet 256x192 weights with prefix fixups
- BGR->RGB normalization, AlphaPose affine, per-joint decode
- Returns IMAGE-SPACE keypoints [K,3] (x,y,conf)
- Fails fast if SPPE not loaded or heatmaps are all-zero
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import sys
import numpy as np
import cv2
import torch
import torch.nn as nn

# ---- Paths to your AlphaPose clone + model files ----
REPO_ROOT = Path("/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline").resolve()
ALPHAPOSE_DIR = REPO_ROOT / "AlphaPose"

# Your confirmed files:
POSE_CFG  = ALPHAPOSE_DIR / "configs/coco/hrnet/256x192_w32_lr1e-3.yaml"
POSE_CKPT = ALPHAPOSE_DIR / "pretrained_models/pose_hrnet_w32_256x192.pth"

def _inject_paths() -> None:
    for p in (REPO_ROOT, ALPHAPOSE_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
_inject_paths()

# ---- AlphaPose utilities ----
def _import_transforms():
    from alphapose.utils.transforms import get_affine_transform, transform_preds  # type: ignore
    return get_affine_transform, transform_preds

def _import_get_max_preds():
    try:
        from alphapose.utils.metrics import get_max_preds  # type: ignore
        return get_max_preds
    except Exception:
        def get_max_preds(hm: torch.Tensor):
            N, K, H, W = hm.shape
            flat = hm.reshape(N, K, -1)
            maxvals, idx = torch.max(flat, dim=2, keepdim=True)
            idx = idx.to(torch.float32)
            preds = torch.zeros((N, K, 2), device=hm.device, dtype=torch.float32)
            preds[..., 0] = (idx % W).squeeze(-1)
            preds[..., 1] = (idx // W).squeeze(-1)
            return preds, maxvals
        return get_max_preds

get_affine_transform, transform_preds = _import_transforms()
get_max_preds = _import_get_max_preds()

@dataclass
class SPPEConfig:
    # HRNet input (H, W) for 256x192 models
    input_size: Tuple[int, int] = (256, 192)
    device: str = "cuda"
    sppe_cfg_yaml: Optional[Path] = POSE_CFG
    sppe_checkpoint: Optional[Path] = POSE_CKPT
    fp16: bool = False

class AlphaPoseRunner:
    """
    API:
      pose_on_boxes(frame_bgr, boxes_xyxy) -> (list[(K,3)], list[float])
      Returns Kx3 [x,y,conf] in ORIGINAL IMAGE COORDS.
    """
    def __init__(self, cfg: Optional[SPPEConfig] = None, sppe_model: Optional[nn.Module] = None):
        self.cfg = cfg or SPPEConfig()
        use_cuda = torch.cuda.is_available() and "cuda" in self.cfg.device
        self.device = torch.device(self.cfg.device if use_cuda else "cpu")
        self.in_h, self.in_w = self.cfg.input_size  # (H,W)
        self.K = 17

        # AlphaPose normalization (expects RGB)
        self.mean = np.array([0.406, 0.457, 0.485], dtype=np.float32)
        self.std  = np.array([0.225, 0.224, 0.229], dtype=np.float32)

        if sppe_model is not None:
            self.model = sppe_model
        else:
            self.model = self._build_from_cfg_dynamic()

        self.model.to(self.device).eval()
        if self.cfg.fp16 and self.device.type == "cuda":
            self.model.half()

    # ----------- dynamic model builder (handles multiple AlphaPose APIs) -----------
    def _build_from_cfg_dynamic(self) -> nn.Module:
        cfgp, ckptp = self.cfg.sppe_cfg_yaml, self.cfg.sppe_checkpoint
        if not cfgp or not ckptp or (not Path(cfgp).is_file()) or (not Path(ckptp).is_file()):
            raise RuntimeError(
                f"AlphaPose cfg/ckpt missing.\n  cfg: {cfgp}\n  ckpt: {ckptp}\n"
                "Provide valid files to run SPPE."
            )

        from alphapose.utils.config import update_config  # type: ignore
        ap_cfg = update_config(str(cfgp))

        # Try multiple builder signatures
        model = None
        try:
            from alphapose.models import builder as B  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Cannot import alphapose.models.builder: {e}")

        # 1) Preferred in many versions: build_sppe(cfg.MODEL, cfg.DATA_PRESET)
        if model is None and hasattr(B, "build_sppe"):
            try:
                model = B.build_sppe(ap_cfg.MODEL, ap_cfg.DATA_PRESET)
            except Exception as e:
                print(f"[AlphaPose] build_sppe failed: {e}")

        # 2) Older/newer variants: get_pose_net(cfg, is_train=False)
        if model is None and hasattr(B, "get_pose_net"):
            try:
                model = B.get_pose_net(ap_cfg, is_train=False)
            except Exception as e:
                print(f"[AlphaPose] get_pose_net failed: {e}")

        # 3) Generic factory: get_model('pose', cfg, ...)
        if model is None and hasattr(B, "get_model"):
            try:
                model = B.get_model('pose', ap_cfg, is_train=False)
            except Exception as e:
                print(f"[AlphaPose] get_model('pose', ...) failed: {e}")

        if model is None:
            raise RuntimeError(
                "Could not build SPPE from alphapose.models.builder. "
                "Your AlphaPose version lacks build_sppe/get_pose_net/get_model for pose."
            )

        # Load weights with prefix fixups
        state = torch.load(str(ckptp), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        def try_load(m, sd):
            missing, unexpected = m.load_state_dict(sd, strict=False)
            return missing, unexpected

        def flip_prefix(sd, prefix: str):
            if all(k.startswith(prefix) for k in sd.keys()):
                return {k[len(prefix):]: v for k, v in sd.items()}
            else:
                return {f"{prefix}{k}": v for k, v in sd.items()}

        # attempt 1: as-is
        missing, unexpected = try_load(model, state)
        # if obviously mismatched, try removing/adding common prefixes
        if len(missing) > 500:
            for pref in ("module.", "model."):
                fixed = flip_prefix(state, pref)
                missing, unexpected = try_load(model, fixed)
                if len(missing) < 500:
                    state = fixed
                    break

        total = sum(p.numel() for p in model.parameters())
        loaded = sum(v.numel() for k, v in model.state_dict().items() if k not in missing)
        print(f"[AlphaPose] Weights loaded. Missing={len(missing)} Unexpected={len(unexpected)} "
              f"ParamsLoaded≈{loaded}/{total}")
        if loaded < total * 0.6:
            raise RuntimeError(
                "Too few weights matched the model. "
                "Check YAML/ckpt pairing (w32 vs w48, 256x192 vs 384x288) and prefixes."
            )
        return model

    # ---------------------------- preprocessing (affine) ----------------------------
    def _preproc(self, img_bgr: np.ndarray, box: np.ndarray):
        x1, y1, x2, y2 = box.astype(np.float32)
        w = max(1.0, float(x2 - x1)); h = max(1.0, float(y2 - y1))
        center = np.array([x1 + 0.5 * w, y1 + 0.5 * h], dtype=np.float32)
        scale = max(w, h) * 1.25  # AlphaPose convention

        getT = get_affine_transform
        trans = getT(center, scale, 0, (self.in_w, self.in_h))  # size=(W,H)

        crop = cv2.warpAffine(img_bgr, trans, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        ten = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
        return ten.to(self.device), trans

    # ----------------------------------- API -----------------------------------
    @torch.inference_mode()
    def pose_on_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: Union[List[List[float]], np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        keypoints_out: List[np.ndarray] = []
        pose_scores_out: List[float] = []

        if boxes_xyxy is None:
            return keypoints_out, pose_scores_out
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return keypoints_out, pose_scores_out

        batch, trans_list = [], []
        for b in boxes:
            ten, trans = self._preproc(frame_bgr, b)
            batch.append(ten); trans_list.append(trans)
        inp = torch.cat(batch, dim=0)  # (N,3,H,W)
        if self.cfg.fp16 and self.device.type == "cuda":
            inp = inp.half()

        hm = self.model(inp)  # (N,K,h,w) or list/tuple
        if isinstance(hm, (list, tuple)):
            hm = hm[-1]
        hm = hm.float()

        if torch.allclose(hm, torch.zeros_like(hm)):
            raise RuntimeError("SPPE produced all-zero heatmaps. Check cfg/ckpt and preprocessing (RGB).")

        preds_hm, maxvals = get_max_preds(hm.detach())  # (N,K,2), (N,K,1)
        N, K, _ = preds_hm.shape
        _, _, Hh, Wh = hm.shape

        # heatmap -> input space
        sx = float(self.in_w) / float(Wh)
        sy = float(self.in_h) / float(Hh)
        preds_in = preds_hm.clone()
        preds_in[..., 0] *= sx
        preds_in[..., 1] *= sy

        # input -> image space via inverse affine
        for i in range(N):
            pts_in = preds_in[i].cpu().numpy()  # (K,2)
            invT = cv2.invertAffineTransform(trans_list[i])
            xy = np.empty((K, 2), dtype=np.float32)
            xy[:, 0] = invT[0,0]*pts_in[:,0] + invT[0,1]*pts_in[:,1] + invT[0,2]
            xy[:, 1] = invT[1,0]*pts_in[:,0] + invT[1,1]*pts_in[:,1] + invT[1,2]

            conf = maxvals[i].cpu().numpy().reshape(-1)
            kpt = np.concatenate([xy, conf[:, None]], axis=1)  # image-space
            keypoints_out.append(kpt)

            vis = conf > 0
            pose_scores_out.append(float(conf[vis].mean() if vis.any() else conf.mean()))
        return keypoints_out, pose_scores_out
