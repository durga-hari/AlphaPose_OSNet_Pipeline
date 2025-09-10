#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
alphapose_wrapper.py (updated)

- Still supports internal detection via AlphaPose YOLOv3.
- NEW: pose_on_boxes(frame, boxes_xyxy) to run HRNet on *external* detections (e.g., YOLOv8).
"""

from __future__ import annotations
import sys, logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

log = logging.getLogger(__name__)

class AlphaPoseNotReady(Exception): pass

class AlphaPoseRunner:
    def __init__(self,
        alphapose_dir: Path,
        device: str = "cuda",
        yolo_cfg: Optional[Path] = None,
        yolo_weights: Optional[Path] = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        pose_cfg: Optional[Path] = None,
        pose_ckpt: Optional[Path] = None,
    ):
        self.alphapose_dir = Path(alphapose_dir).resolve()
        if not self.alphapose_dir.exists(): raise FileNotFoundError(self.alphapose_dir)
        if str(self.alphapose_dir) not in sys.path: sys.path.insert(0, str(self.alphapose_dir))

        import torch
        wants_cuda = (str(device).lower() == "cuda")
        self.device = "cuda" if wants_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and wants_cuda: log.warning("CUDA requested but unavailable; using CPU.")
        self._torch_device = torch.device(self.device)

        # smoke-test package
        try:
            from alphapose.models import builder as _
            from alphapose.utils.config import update_config as _
        except Exception as e:
            raise AlphaPoseNotReady("Failed to import alphapose package.") from e

        # YOLOv3 (optional)
        self.yolo_cfg = Path(yolo_cfg) if yolo_cfg else None
        self.yolo_weights = Path(yolo_weights) if yolo_weights else None
        self.yolo_conf = float(yolo_conf); self.yolo_iou = float(yolo_iou)
        self._det = None

        # HRNet
        self.pose_cfg = Path(pose_cfg) if pose_cfg else None
        self.pose_ckpt = Path(pose_ckpt) if pose_ckpt else None
        self._sppe = None; self._pose_cfg = None

    def warmup(self, enable_internal_detector: bool = True):
        # Optional internal YOLOv3
        if enable_internal_detector and self.yolo_cfg and self.yolo_weights:
            try:
                from alphapose.detector.yolo_api import YOLODetector
                self._det = YOLODetector(
                    cfgfile=str(self.yolo_cfg), weightfile=str(self.yolo_weights),
                    conf_thres=self.yolo_conf, nms_thres=self.yolo_iou, device=self.device
                )
                log.info("AlphaPose internal detector (YOLOv3) loaded.")
            except Exception as e:
                self._det = None; log.warning("Internal detector failed: %s", e)
        else:
            self._det = None

        # HRNet (SPPE)
        try:
            import torch
            from alphapose.models import builder
            from alphapose.utils.config import update_config

            if not self.pose_cfg or not self.pose_cfg.exists(): raise FileNotFoundError(self.pose_cfg)
            if not self.pose_ckpt or not self.pose_ckpt.exists(): raise FileNotFoundError(self.pose_ckpt)

            cfg = update_config(str(self.pose_cfg))
            self._pose_cfg = cfg
            self._sppe = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
            ckpt = torch.load(str(self.pose_ckpt), map_location=self._torch_device)
            state_dict = ckpt.get("state_dict", ckpt)
            self._sppe.load_state_dict(state_dict, strict=False)
            self._sppe.to(self._torch_device).eval()
            log.info("HRNet (SPPE) loaded.")
        except Exception as e:
            self._sppe = None; self._pose_cfg = None; log.warning("HRNet failed: %s", e)

    # --- Internal detector path (legacy) -------------------------------------
    def _internal_detect(self, frame_bgr) -> List[List[float]]:
        if self._det is None: return []
        import torch
        H, W = frame_bgr.shape[:2]
        try:
            prepped = self._det.image_preprocess(frame_bgr)
        except Exception:
            prepped = self._det.image_preprocess([frame_bgr])

        im_batches, im_dim_list = None, None
        if isinstance(prepped, (torch.Tensor, np.ndarray)):
            im_batches = prepped; im_dim_list = [(W, H)]
        elif isinstance(prepped, (list, tuple)):
            if len(prepped) == 2: im_batches, im_dim_list = prepped
            elif len(prepped) >= 3: im_batches, im_dim_list = prepped[0], prepped[2] if isinstance(prepped[2], list) else [(W,H)]
        else:
            im_batches = prepped; im_dim_list = [(W, H)]
        if not isinstance(im_dim_list, (list, tuple)):
            im_dim_list = [(W, H)]
        if not hasattr(im_dim_list, "shape"):
            im_dim_list = torch.tensor(im_dim_list, dtype=torch.float32)

        try:
            det_out = self._det.images_detection(im_batches, im_dim_list)
        except TypeError:
            det_out = self._det.images_detection(im_batches, im_dim_list, None)

        dets = det_out[0] if isinstance(det_out, (list, tuple)) and len(det_out) > 0 else det_out
        if dets is None: return []
        boxes = []
        try:
            for d in dets:
                x1, y1, x2, y2 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                conf = float(d[4]) if len(d) > 4 else 1.0
                boxes.append([x1, y1, x2, y2, conf])
        except Exception:
            boxes = []
        return boxes

    # --- Pose on given boxes (works with YOLOv8) -----------------------------
    def pose_on_boxes(self, frame_bgr, boxes_xyxy: List[List[float]]):
        """Run HRNet on provided [x1,y1,x2,y2,(score?)] boxes."""
        if self._sppe is None or not boxes_xyxy: return [], []
        import torch
        from alphapose.utils.transforms import get_affine_transform

        cfg = self._pose_cfg
        inp_w, inp_h = (256, 192)
        if hasattr(cfg.DATA_PRESET, "IMAGE_SIZE"):
            inp_w, inp_h = cfg.DATA_PRESET.IMAGE_SIZE  # (w,h)

        results, scores = [], []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            center = np.array([x1 + 0.5*w, y1 + 0.5*h], dtype=np.float32)
            scale  = max(h/200.0, w/200.0)
            trans  = get_affine_transform(center, scale, 0, (int(inp_w), int(inp_h)))
            patch  = cv2.warpAffine(frame_bgr, trans, (int(inp_w), int(inp_h)))
            patch  = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            inp    = np.transpose(patch, (2,0,1))[None, ...]
            inp_t  = torch.from_numpy(inp).to(self._torch_device)
            with torch.no_grad():
                hm = self._sppe(inp_t)  # [1,K,H,W]
            hm_np = hm.squeeze(0).cpu().numpy()
            K, Hm, Wm = hm_np.shape

            kpts, kscs = [], []
            for j in range(K):
                idx = int(np.argmax(hm_np[j])); yy, xx = divmod(idx, Wm)
                score = float(np.max(hm_np[j]))
                sx, sy = w / Wm, h / Hm
                kpts.append((float(x1 + xx*sx), float(y1 + yy*sy), score))
                kscs.append(score)
            results.append(kpts)
            scores.append(float(np.mean(kscs)) if kscs else 0.0)
        return results, scores

    def infer(self, frame_bgr, use_internal_detector: bool = True):
        """
        Convenience path:
         - if use_internal_detector: run internal detect + pose
         - else: just return empty (for external detector mode)
        """
        if use_internal_detector:
            boxes_full = self._internal_detect(frame_bgr)  # [x1,y1,x2,y2,score]
            boxes = [[b[0], b[1], b[2], b[3]] for b in boxes_full]
            kpts, pose_scores = self.pose_on_boxes(frame_bgr, boxes_full)
            n = min(len(boxes), len(kpts), len(pose_scores))
            boxes, kpts, pose_scores = boxes[:n], kpts[:n], pose_scores[:n]
            det_scores = [b[4] for b in boxes_full][:n] if boxes_full else []
            scores = pose_scores if any(pose_scores) else (det_scores if det_scores else [0.0]*n)
            feats  = [None]*n
            return boxes, kpts, scores, feats
        else:
            return [], [], [], []
