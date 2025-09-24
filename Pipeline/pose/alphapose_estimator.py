from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2
import sys


sys.path.append("/home/arun_remote/DaRA_Thesis/AlphaPose_OSNet_Pipeline/AlphaPose")


@dataclass
class AlphaPoseCfg:
    input_size: Tuple[int, int] = (384,288) #(256, 192) # (H,W)
    device: str = "cuda"
    cfg_yaml: Optional[str | Path] = None
    checkpoint: Optional[str | Path] = None
    fp16: bool = False


class AlphaPoseEstimator:
    """pose_on_boxes(frame_bgr, boxes_xyxy)->(list[(K,3)], list[float])"""
    def __init__(self, config: AlphaPoseCfg | None = None):
        import torch
        self.cfg = config or AlphaPoseCfg()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() and "cuda" in self.cfg.device else "cpu")
        self.in_h, self.in_w = self.cfg.input_size
        self.mean = np.array([0.406, 0.457, 0.485], dtype=np.float32)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
        self.model = self._build()
        self.model.to(self.device).eval()
        if self.cfg.fp16 and self.device.type == "cuda":
            self.model.half()


    # --- AlphaPose imports ---
    def _ap_transforms(self):
        from alphapose.utils.transforms import get_affine_transform, transform_preds # noqa: F401
        return get_affine_transform


    def _ap_get_max_preds(self):
        try:
            from alphapose.utils.metrics import get_max_preds # type: ignore
            return get_max_preds
        except Exception:
            import torch
            def get_max_preds(hm: "torch.Tensor"):
                N, K, H, W = hm.shape
                flat = hm.reshape(N, K, -1)
                maxvals, idx = torch.max(flat, dim=2, keepdim=True)
                idx = idx.to(torch.float32)
                preds = torch.zeros((N, K, 2), device=hm.device, dtype=torch.float32)
                preds[..., 0] = (idx % W).squeeze(-1)
                preds[..., 1] = (idx // W).squeeze(-1)
                return preds, maxvals
            return get_max_preds


    def _build(self):
        import torch
        from alphapose.utils.config import update_config
        from alphapose.models import builder as B
        if not self.cfg.cfg_yaml or not self.cfg.checkpoint:
            raise RuntimeError("AlphaPose cfg/checkpoint required")
        ap_cfg = update_config(str(self.cfg.cfg_yaml))
        try:
            model = B.build_sppe(ap_cfg['MODEL'], ap_cfg['DATA_PRESET'])
        except Exception:
            model = B.get_pose_net(ap_cfg, is_train=False)
        state = torch.load(str(self.cfg.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model


    def _preproc(self, img_bgr: np.ndarray, box: np.ndarray):
        import torch
        get_affine_transform = self._ap_transforms()
        x1, y1, x2, y2 = box.astype(np.float32)
        w = max(1.0, float(x2 - x1)); h = max(1.0, float(y2 - y1))
        center = np.array([x1 + 0.5 * w, y1 + 0.5 * h], dtype=np.float32)
        scale = max(w, h) * 1.35  #BBox scale prior value 1.25
        trans = get_affine_transform(center, scale, 0, (self.in_w, self.in_h))
        crop = cv2.warpAffine(img_bgr, trans, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - self.mean) / self.std
        ten = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0)
        return ten.to(self.device), trans


    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        import torch
        get_max_preds = self._ap_get_max_preds()
        keypoints_out: List[np.ndarray] = []
        pose_scores_out: List[float] = []
        heatmaps_out: list[np.ndarray] = []
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return keypoints_out, pose_scores_out
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        batch, trans_list = [], []
        for b in boxes:
            ten, trans = self._preproc(frame_bgr, b)
            batch.append(ten); trans_list.append(trans)
        inp = torch.cat(batch, dim=0)
        if self.cfg.fp16 and self.device.type == "cuda":
            inp = inp.half()
        hm = self.model(inp)
        if isinstance(hm, dict):
            hm = hm.get("heatmaps", hm.get("hm", hm.get("output", hm)))
        if isinstance(hm, (list, tuple)):
            hm = hm[-1]
        hm = hm.float()
        preds_hm, _ = get_max_preds(hm.detach())
        hm_conf = hm.sigmoid()
        _, maxvals = get_max_preds(hm_conf.detach()) # (N,K,1)
        N, K, Hh, Wh = hm.shape
        for i in range(N):
            heatmaps_out.append(hm[i].detach().cpu().numpy())
        sx = float(self.in_w) / float(Wh)
        sy = float(self.in_h) / float(Hh)
        preds_in = preds_hm.clone()
        preds_in[..., 0] *= sx
        preds_in[..., 1] *= sy
        for i in range(N):
                pts_in = preds_in[i].cpu().numpy()
                invT = cv2.invertAffineTransform(trans_list[i])
                xy = np.empty((K, 2), dtype=np.float32)
                xy[:, 0] = invT[0,0]*pts_in[:,0] + invT[0,1]*pts_in[:,1] + invT[0,2]
                xy[:, 1] = invT[1,0]*pts_in[:,0] + invT[1,1]*pts_in[:,1] + invT[1,2]
                conf = np.clip(maxvals[i].cpu().numpy().reshape(-1), 0.3, 1.0) #tune this to reduce skeleton noise
                keypoints_out.append(np.c_[xy, conf])
                pose_scores_out.append(float(conf[conf > 0].mean() if (conf > 0).any() else conf.mean()))
        return keypoints_out, pose_scores_out,heatmaps_out