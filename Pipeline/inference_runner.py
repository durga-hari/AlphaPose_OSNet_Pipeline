# import os
# import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ALPHAPOSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../AlphaPose"))
# sys.path.insert(0, ALPHAPOSE_DIR)
# sys.path.insert(0, os.path.join(ALPHAPOSE_DIR, "alphapose"))



# import torch
# from alphapose.utils.transforms import flip, flip_heatmap


# def run_inference(model, inps, device, flip_test=False, joint_pairs=None):
#     """
#     Run pose inference for a batch of images.

#     Args:
#         model: The AlphaPose pose model (SPPE)
#         inps: Preprocessed input tensor (B x C x H x W)
#         device: torch.device
#         flip_test: Whether to apply flip testing
#         joint_pairs: Joint pairs for flip heatmap alignment

#     Returns:
#         heatmaps: Output heatmaps (B x num_joints x H x W)
#     """
#     model.eval()
#     inps = inps.to(device)
#     batch_size = inps.size(0)

#     with torch.no_grad():
#         if flip_test:
#             assert joint_pairs is not None, "flip_test requires joint_pairs"
#             flipped_inps = flip(inps)
#             inps_cat = torch.cat((inps, flipped_inps), dim=0)
#             out = model(inps_cat)

#             out_orig = out[0:batch_size]
#             out_flip = flip_heatmap(
#                 out[batch_size:], joint_pairs, shift=True
#             )
#             heatmaps = (out_orig + out_flip) / 2
#         else:
#             heatmaps = model(inps)

#     return heatmaps.cpu()
# AlphaPose_OSNet_Pipeline_Optimized/pipeline/inference_runner.py
# AlphaPose_OSNet_Pipeline_Optimized/pipeline/inference_runner.py
import os
import sys
from typing import Tuple, List, Union, Optional

import numpy as np
import torch
import cv2

# Optional: help resolve AlphaPose if your layout needs it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHAPOSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../AlphaPose"))
if os.path.isdir(ALPHAPOSE_DIR):
    sys.path.insert(0, ALPHAPOSE_DIR)
    sys.path.insert(0, os.path.join(ALPHAPOSE_DIR, "alphapose"))

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _model_device(model: torch.nn.Module) -> torch.device:
    try: return next(model.parameters()).device
    except Exception: return torch.device("cpu")

def _infer_input_size_from_name(model: torch.nn.Module) -> Tuple[int, int]:
    # Prefer HRNet defaults; common fallbacks included
    name = getattr(model, "__class__", type(model)).__name__.lower()
    text = name + " " + " ".join([n for n in dir(model) if isinstance(n, str)])
    for h, w in [(384, 288), (256, 192), (512, 384), (320, 256)]:
        if str(h) in text and str(w) in text:
            return (h, w)
    return (384, 288)

def _xyxy_from_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be length-4")
    x1, y1, x2, y2 = bbox
    # If looks like XYXY, keep; else treat as XYWH
    if x2 > x1 and y2 > y1:
        return float(x1), float(y1), float(x2), float(y2)
    return float(x1), float(y1), float(x1 + x2), float(y1 + y2)

def _crop_and_resize(img: np.ndarray, bbox_xyxy: Tuple[float,float,float,float],
                     dst_hw: Tuple[int,int]) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(W, int(round(x2))), min(H, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        xi1, yi1, xi2, yi2 = 0, 0, W, H
    crop = img[yi1:yi2, xi1:xi2]
    dst_h, dst_w = dst_hw
    if crop.size == 0:
        crop = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    return crop, (xi1, yi1, xi2, yi2)

def _to_tensor_nchw(img_rgb: np.ndarray) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)  # 1×C×H×W

def _pick_heatmap_tensor(out) -> Optional[torch.Tensor]:
    # Accept a variety of model outputs and return a 4D heatmap tensor (B,J,Hh,Wh) if possible
    if isinstance(out, torch.Tensor) and out.ndim == 4:
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                return x
    if isinstance(out, dict):
        for k in ("heatmaps", "hm", "heatmap", "out", "preds"):
            x = out.get(k, None)
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                return x
    return None

def _pick_keypoints_direct(out) -> Optional[np.ndarray]:
    # If model already returned keypoints
    # Accept shapes: (J,2), (J,3) or batched forms
    if isinstance(out, dict) and "keypoints" in out:
        kp = out["keypoints"]
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 2 and kp.shape[1] in (2, 3):
            return kp
        if kp.ndim == 3 and kp.shape[-1] in (2, 3):
            return kp[0]
    if isinstance(out, (list, tuple)):
        try:
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] in (2, 3):
                return arr
            if arr.ndim == 3 and arr.shape[-1] in (2, 3):
                return arr[0]
        except Exception:
            pass
    if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] in (2, 3):
        return out.astype(np.float32)
    return None

def _decode_heatmaps(hm: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int]]:
    hm = hm.detach().cpu().numpy()
    B, J, Hh, Wh = hm.shape
    if B < 1: raise ValueError("empty batch")
    hm0 = hm[0]
    flat = hm0.reshape(J, -1)
    idx = flat.argmax(axis=1)
    ys = (idx // Wh).astype(np.float32)
    xs = (idx %  Wh).astype(np.float32)
    maxv = flat.max(axis=1).astype(np.float32)
    coords = np.stack([xs, ys], axis=1)  # (J,2) heatmap coords
    return coords, maxv, (Hh, Wh)

def _map_to_img(coords_xy: np.ndarray, maxvals: np.ndarray,
                crop_xyxy: Tuple[int,int,int,int],
                heatmap_hw: Tuple[int,int],
                input_hw: Tuple[int,int]) -> List[List[float]]:
    xi1, yi1, xi2, yi2 = crop_xyxy
    Hh, Wh = heatmap_hw
    # Scale per heatmap coordinate:
    kx = (xi2 - xi1) / float(max(1, Wh))
    ky = (yi2 - yi1) / float(max(1, Hh))
    out = []
    for (xh, yh), sc in zip(coords_xy, maxvals):
        x_img = xi1 + xh * kx
        y_img = yi1 + yh * ky
        out.append([float(x_img), float(y_img), float(sc)])
    return out

def _forward_model(model, inp: torch.Tensor):
    # Try plain forward; if that fails, try a few common kwargs
    try:
        return model(inp)
    except Exception:
        try:
            return model(inp, return_heatmaps=True)
        except Exception:
            try:
                return model(inp, get_heatmap=True)
            except Exception:
                return None

def _pose_from_one_color(model, img_color: np.ndarray, bbox_xyxy: Tuple[float,float,float,float]) -> Optional[dict]:
    H_infer, W_infer = _infer_input_size_from_name(model)
    crop, crop_xyxy = _crop_and_resize(img_color, bbox_xyxy, (H_infer, W_infer))
    inps = _to_tensor_nchw(crop).to(_model_device(model))
    model.eval()
    with torch.no_grad():
        out = _forward_model(model, inps)
    if out is None:
        return None

    # Case A: direct keypoints
    kpd = _pick_keypoints_direct(out)
    if kpd is not None:
        if kpd.shape[1] == 2:
            kpd = np.hstack([kpd, np.ones((kpd.shape[0], 1), dtype=np.float32)])
        # Map resized-crop coords -> image coords
        xs = kpd[:, 0] / max(1e-6, W_infer) * (crop_xyxy[2] - crop_xyxy[0]) + crop_xyxy[0]
        ys = kpd[:, 1] / max(1e-6, H_infer) * (crop_xyxy[3] - crop_xyxy[1]) + crop_xyxy[1]
        kps = [[float(x), float(y), float(s)] for x, y, s in zip(xs, ys, kpd[:, 2])]
        return {"keypoints": kps}

    # Case B: heatmaps -> decode
    hm = _pick_heatmap_tensor(out)
    if hm is not None:
        coords_xy, maxv, hw = _decode_heatmaps(hm)
        kps = _map_to_img(coords_xy, maxv, crop_xyxy, hw, (H_infer, W_infer))
        return {"keypoints": kps}

    return None  # unknown structure

def run_inference(model: torch.nn.Module,
                  img_rgb: np.ndarray,
                  bbox: Union[List[float], Tuple[float,float,float,float]]) -> Optional[dict]:
    """
    Accepts RGB frame + bbox (XYWH or XYXY). Tries RGB, then BGR automatically.
    Returns {"keypoints": [[x,y,score], ...]} in original image coords, or None on failure.
    """
    if img_rgb is None or not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3:
        return None
    x1, y1, x2, y2 = _xyxy_from_bbox(list(map(float, bbox)))

    # Try RGB path
    out = _pose_from_one_color(model, img_rgb, (x1, y1, x2, y2))
    if out is not None:
        return out
    # Fallback: try BGR path
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = _pose_from_one_color(model, img_bgr, (x1, y1, x2, y2))
    return out





