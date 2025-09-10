#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import List, Tuple, Optional

from ..config import (
    ALPHAPOSE_DIR, YOLO_CFG, YOLO_WEIGHTS, YOLO_CONF, YOLO_IOU,
    Y8_MODEL, Y8_WEIGHTS, Y8_CONF, Y8_IOU,
    POSE_CFG, POSE_CKPT, OSNET_WEIGHT, OUTPUT_ROOT, DEVICE,
    REID_SIM_THR, REID_IOU_THR, REID_TTL, DETECTOR_DEFAULT
)
from AlphaPose_OSNet_Pipeline.utils.io_utils import ensure_dir
from .alphapose_wrapper import AlphaPoseRunner, AlphaPoseNotReady
from .osnet_wrapper import OSNetExtractor
from .strongsort_tracker import StrongSORTTracker
from .inference_runner import InferenceRunner
from .yolov8_wrapper import YOLOv8Detector

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    p = argparse.ArgumentParser(description="Toggle: AlphaPose YOLOv3 or YOLOv8 + HRNet + OSNet")
    p.add_argument("--video", required=True, help="Video file or directory.")
    p.add_argument("--cams", "--cam-id", dest="cams", default=None,
                   help="Camera ID or comma-separated list. Defaults to video stem(s).")
    p.add_argument("--out", default=str(OUTPUT_ROOT), help="Output directory root.")
    p.add_argument("--alphapose-path", default=str(ALPHAPOSE_DIR), help="Path to AlphaPose folder.")
    p.add_argument("--device", choices=["cuda", "cpu"], default=DEVICE)
    p.add_argument("--draw", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-embeds", action="store_true", help="Include OSNet embedding in JSONL.")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)

    # Detector toggle
    p.add_argument("--detector", choices=["alphapose","yolov8"], default=DETECTOR_DEFAULT)
    # YOLOv8 knobs
    p.add_argument("--y8-model", default=(str(Y8_WEIGHTS) if Y8_WEIGHTS.exists() else Y8_MODEL))
    p.add_argument("--y8-conf", type=float, default=Y8_CONF)
    p.add_argument("--y8-iou",  type=float, default=Y8_IOU)
    # AlphaPose YOLOv3 knobs
    p.add_argument("--yolo-cfg", default=str(YOLO_CFG))
    p.add_argument("--yolo-weights", default=str(YOLO_WEIGHTS))
    p.add_argument("--yolo-conf", type=float, default=YOLO_CONF)
    p.add_argument("--yolo-iou",  type=float, default=YOLO_IOU)
    # Pose + ReID
    p.add_argument("--pose-cfg",  default=str(POSE_CFG))
    p.add_argument("--pose-ckpt", default=str(POSE_CKPT))
    p.add_argument("--osnet-weight", default=(str(OSNET_WEIGHT) if OSNET_WEIGHT.exists() else None))
    # StrongSORT
    p.add_argument("--reid-sim-thr", type=float, default=REID_SIM_THR)
    p.add_argument("--reid-iou-thr", type=float, default=REID_IOU_THR)
    p.add_argument("--reid-ttl",     type=int,   default=REID_TTL)
    return p.parse_args()

def main():
    args = parse_args()

    in_path  = Path(args.video)
    out_root = Path(args.out).resolve()
    alpha_dir = Path(args.alphapose_path).resolve()
    ensure_dir(out_root)

    # HRNet & (optional) internal YOLOv3
    try:
        ap = AlphaPoseRunner(
            alphapose_dir=alpha_dir, device=args.device,
            yolo_cfg=(Path(args.yolo_cfg) if args.detector=="alphapose" else None),
            yolo_weights=(Path(args.yolo_weights) if args.detector=="alphapose" else None),
            yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou,
            pose_cfg=Path(args.pose_cfg), pose_ckpt=Path(args.pose_ckpt),
        )
        ap.warmup(enable_internal_detector=(args.detector=="alphapose"))
    except (AlphaPoseNotReady, Exception) as e:
        log.error("AlphaPose init failed: %s", e); return

    # YOLOv8 (only if selected)
    y8 = None
    if args.detector == "yolov8":
        # Allow either a .pt path or a model id
        y8_weights = Path(args.y8_model) if args.y8_model.endswith(".pt") else None
        y8_model_id = (args.y8_model if not args.y8_model.endswith(".pt") else "yolov8s")
        try:
            y8 = YOLOv8Detector(model=y8_model_id, weights=y8_weights,
                                device=args.device, conf=args.y8_conf, iou=args.y8_iou)
        except ImportError as e:
            log.error("%s", e); return
        except Exception as e:
            log.error("YOLOv8 init failed: %s", e); return

    # OSNet ReID
    osn = OSNetExtractor(weight=(Path(args.osnet_weight) if args.osnet_weight else None), device=args.device)
    if not osn.is_ready():
        log.warning("OSNet disabled (missing torchreid or weight). Proceeding without embeddings.")

    tracker = StrongSORTTracker(sim_thr=args.reid_sim_thr, iou_thr=args.reid_iou_thr, ttl=args.reid_ttl)

    runner = InferenceRunner(
        alphapose=ap, osnet=osn, tracker=tracker,
        detector=args.detector, y8=y8,
        draw=bool(args.draw), save_embeds=bool(args.save_embeds),
    )

    # Jobs
    jobs: List[Tuple[Path, str]] = []
    if in_path.is_dir():
        vids = sorted([p for p in in_path.iterdir() if p.suffix.lower() in {".mp4",".avi",".mov",".mkv"}])
        if args.cams:
            cam_list = [c.strip() for c in args.cams.split(",") if c.strip()]
            for i, v in enumerate(vids):
                cam_id = cam_list[i] if i < len(cam_list) else v.stem
                jobs.append((v, cam_id))
        else:
            for v in vids: jobs.append((v, v.stem))
    else:
        jobs.append((in_path, (args.cams or in_path.stem)))

    for vpath, cam in jobs:
        cam_out = out_root / cam; ensure_dir(cam_out)
        runner.run_on_video(
            video_path=vpath, out_dir=cam_out, cam_id=cam,
            save_video=bool(args.save_video), save_json=bool(args.save_json),
            start_frame=args.start_frame, max_frames=args.max_frames,
        )

if __name__ == "__main__":
    main()
