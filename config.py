#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# --- Repo layout --------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent
ALPHAPOSE_DIR = (REPO_ROOT / "AlphaPose").resolve()
PIPELINE_DIR  = (REPO_ROOT / "Pipeline").resolve()

# --- IO defaults --------------------------------------------------------------
INPUT_ROOT  = Path("/home/athena/DaRA_Thesis/Session 4").resolve()
OUTPUT_ROOT = Path("/home/athena/DaRA_Thesis/Output_AP_OSNET").resolve()

# --- Detector selection -------------------------------------------------------
# "alphapose" -> AlphaPose YOLOv3 (works without Ultralytics)
# "yolov8"    -> Ultralytics YOLOv8 (requires: pip install ultralytics)
DETECTOR_DEFAULT = "alphapose"

# --- AlphaPose YOLOv3 (internal) ---------------------------------------------
YOLO_CFG     = (ALPHAPOSE_DIR / "detector/yolo/cfg/yolov3.cfg").resolve()
YOLO_WEIGHTS = (ALPHAPOSE_DIR / "detector/yolo/weight/yolov3.weights").resolve()
YOLO_CONF    = 0.25
YOLO_IOU     = 0.45

# --- YOLOv8 (Ultralytics) -----------------------------------------------------
Y8_MODEL   = "yolov8s"  # used only if Y8_WEIGHTS doesn't exist
Y8_WEIGHTS = (PIPELINE_DIR / "pretrained/detectors/yolov8s.pt").resolve()
Y8_CONF    = 0.25
Y8_IOU     = 0.45

# --- Pose: HRNet (COCO-17) ----------------------------------------------------
POSE_CFG  = (ALPHAPOSE_DIR / "configs/coco/hrnet/256x192_w32_lr1e-3.yaml").resolve()
POSE_CKPT = (ALPHAPOSE_DIR / "pretrained_models/pose_hrnet_w32_256x192.pth").resolve()

# --- ReID: OSNet --------------------------------------------------------------
OSNET_WEIGHT = (PIPELINE_DIR / "pretrained/osnet_x1_0_msmt17.pth").resolve()

# --- Runtime defaults ---------------------------------------------------------
DEVICE       = "cuda"   # auto-falls back to CPU
REID_SIM_THR = 0.45
REID_IOU_THR = 0.20
REID_TTL     = 60

@dataclass
class ModelPaths:
    alphapose_dir: Path = ALPHAPOSE_DIR
    yolo_cfg: Path = YOLO_CFG
    yolo_weights: Path = YOLO_WEIGHTS
    y8_model: str = Y8_MODEL
    y8_weights: Optional[Path] = Y8_WEIGHTS
    pose_cfg: Path = POSE_CFG
    pose_ckpt: Path = POSE_CKPT
    osnet_weight: Optional[Path] = OSNET_WEIGHT

@dataclass
class RunDefaults:
    input_root: Path = INPUT_ROOT
    output_root: Path = OUTPUT_ROOT
    device: str = DEVICE
    detector: str = DETECTOR_DEFAULT
    yolo_conf: float = YOLO_CONF
    yolo_iou: float = YOLO_IOU
    y8_conf: float = Y8_CONF
    y8_iou: float = Y8_IOU
    reid_sim_thr: float = REID_SIM_THR
    reid_iou_thr: float = REID_IOU_THR
    reid_ttl: int = REID_TTL

def print_config():
    mp, rd = ModelPaths(), RunDefaults()
    print("=== REPO ROOT ===", REPO_ROOT)
    print("\n=== MODEL PATHS ===")
    for k, v in asdict(mp).items(): print(f"{k:12s}: {v}")
    print("\n=== DEFAULTS ===")
    for k, v in asdict(rd).items(): print(f"{k:12s}: {v}")

if __name__ == "__main__":
    print_config()
