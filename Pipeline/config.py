# /home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/Pipeline/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os, re, torch

# Recognized video extensions
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}

def _infer_cam_id(p: Path) -> str:
    """
    Derive a camera id from file name.
    Examples:
      'FC_1_cam1080.mp4' -> 'FC_1'
      'AC_10_sceneA.mkv' -> 'AC_10'
      'LobbyWest.mp4'    -> 'LobbyWest'
    """
    stem = p.stem
    m = re.search(r"(?:^|[_\-\s])((?:FC|AC)_[0-9]+)", stem, flags=re.I)
    if m:
        return m.group(1).upper()
    return stem  # fallback: use filename (without extension)

@dataclass
class Config:
    # ---- Input session folder (with space is OK) ----
    input_dir: str = "/home/athena/DaRA_Thesis/Session 4"

    # Populated automatically from input_dir in __post_init__
    input_video_paths: list[str] = field(default_factory=list)
    camera_ids: list[str]        = field(default_factory=list)

    # ---- Output ----
    output_dir: str = "/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/Output_AP_OSNET"

    # ---- Device / runtime ----
    device: str = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = os.environ.get("FP16", "1").lower() in ("1", "true", "yes")
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    draw_outputs: bool = True

    # ---- AlphaPose base (all other AP subpaths are derived from this) ----
    alphapose_path: str = "/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/AlphaPose"

    # Auto-derived AlphaPose paths (filled in __post_init__)
    detector_config: str = ""
    detector_checkpoint: str = ""
    yolo_names: str = ""
    alphapose_config: str = ""
    alphapose_checkpoint: str = ""

    # ---- OSNet (ReID) ----
    osnet_model: str = "osnet_x1_0"
    osnet_weights: str = "/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/Pipeline/pretrained/osnet_x1_0_msmt17.pth"

    # ---- Thresholds / perf ----
    det_conf_threshold: float = 0.25
    det_nms_threshold: float = 0.45
    frame_skip: int = 1
    detector_scale: float = 1.0
    progress_every: int = 200
    checkpoint_every: int = 1000
    debug_draw_dets: bool = False

    def __post_init__(self):
        # Derive AlphaPose subpaths (same subfolders, new base)
        base = Path(self.alphapose_path)
        self.detector_config      = str(base / "detector/yolo/cfg/yolov3-spp.cfg")
        self.detector_checkpoint  = str(base / "detector/yolo/data/yolov3-spp.weights")
        self.yolo_names           = str(base / "detector/yolo/data/coco.names")
        self.alphapose_config     = str(base / "configs/hrnet/pose_hrnet_w32_384x288.yaml")
        self.alphapose_checkpoint = str(base / "pretrained_models/pose_hrnet_w32_384x288.pth")

        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Discover videos from input_dir (flat first, then recurse if needed)
        in_dir = Path(self.input_dir)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found: {in_dir}")

        vids = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        if not vids:
            vids = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

        if not vids:
            raise FileNotFoundError(f"No video files found under: {in_dir}")

        self.input_video_paths = [str(p) for p in vids]
        self.camera_ids        = [_infer_cam_id(p) for p in vids]

        if len(self.camera_ids) != len(self.input_video_paths):
            raise RuntimeError("camera_ids and input_video_paths length mismatch")

    def validate(self) -> list[str]:
        checks = [
            (self.detector_config, "Detector cfg"),
            (self.detector_checkpoint, "Detector weights"),
            (self.yolo_names, "Detector names"),
            (self.alphapose_config, "AlphaPose YAML"),
            (self.alphapose_checkpoint, "AlphaPose checkpoint"),
            (self.osnet_weights, "OSNet weights"),
        ]
        missing = [f"{label} not found: {p}" for p, label in checks if not Path(p).is_file()]
        for p in self.input_video_paths:
            if not Path(p).is_file():
                missing.append(f"Input video missing: {p}")
        return missing
