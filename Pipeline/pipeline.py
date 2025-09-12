# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

# ---- Our utils (package path you asked for) -------------------------------
from AlphaPose_OSNet_Pipeline.utils.io_utils import (
    ensure_dir,
    safe_stem,
    video_writer,
    jsonl_writer,
)

# ---- Pipeline modules -----------------------------------------------------
# AlphaPose wrapper (no 'device' kwarg!)
from AlphaPose_OSNet_Pipeline.Pipeline.alphapose_wrapper import (
    AlphaPoseRunner,
    SPPEConfig,
)

# OSNet ReID extractor (expects 'weight' kwarg; returns features for tracker)
from AlphaPose_OSNet_Pipeline.Pipeline.osnet_wrapper import OSNetExtractor

# Detection
from AlphaPose_OSNet_Pipeline.Pipeline.yolov8_wrapper import YOLOv8Detector

# Tracking (your provided signature)
from AlphaPose_OSNet_Pipeline.Pipeline.strongsort_tracker import StrongSORTTracker

# Inference runner (your provided signature)
from AlphaPose_OSNet_Pipeline.Pipeline.inference_runner import InferenceRunner


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AlphaPose + OSNet pipeline")

    # IO
    p.add_argument("--video", required=True, type=str, help="Path to input video")
    p.add_argument("--out", required=True, type=str, help="Output directory")

    # Options
    p.add_argument(
        "--detector",
        default="yolov8",
        choices=["alphapose", "yolov8"],
        help="Which detector to use for person boxes",
    )
    p.add_argument("--draw", action="store_true", help="Draw boxes/poses on frames")
    p.add_argument("--save-video", action="store_true", help="Write annotated video")
    p.add_argument("--save-json", action="store_true", help="Write poses JSONL")
    p.add_argument("--save-embeds", action="store_true", help="Save OSNet embeddings")

    # Device (used for non-AlphaPose parts; AlphaPoseRunner pulls device from SPPEConfig)
    p.add_argument("--device", default="cuda", type=str, help="cuda or cpu")

    # YOLOv8 detector knobs
    p.add_argument("--y8-model", type=str, required=False, help="Path to yolov8 *.pt")
    p.add_argument("--y8-imgsz", type=int, default=640)
    p.add_argument("--y8-half", action="store_true")
    p.add_argument("--y8-conf", type=float, default=0.25)
    p.add_argument("--y8-iou", type=float, default=0.45)

    # OSNet weights
    p.add_argument(
        "--osnet-weights",
        type=str,
        default=str(
            Path(__file__).parent / "pretrained" / "osnet_x1_0_msmt17.pth"
        ),
        help="Path to OSNet weights",
    )

    # Misc
    p.add_argument("--max-frames", type=int, default=-1, help="Limit frames for debug")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    p.add_argument("--log-every", type=int, default=50, help="Log every N frames")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    log = logging.getLogger(__name__)
    log.debug("Parsed args: %s", vars(args))

    video_path = Path(args.video)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    base = safe_stem(video_path)
    # Where we’ll write outputs (paths are created here; passing down is OK)
    out_video_path: Optional[Path] = (
        (out_dir / f"{base}_ann.mp4") if args.save_video else None
    )
    out_jsonl_path: Optional[Path] = (
        (out_dir / f"{base}_poses.jsonl") if args.save_json else None
    )
    out_embeds_dir: Optional[Path] = (
        (out_dir / f"{base}_embeds") if args.save_embeds else None
    )
    if out_embeds_dir:
        ensure_dir(out_embeds_dir)

    # ------------------ Build components ------------------
    # AlphaPose Runner (device chosen inside via SPPEConfig; no 'device' kwarg)

    ap = AlphaPoseRunner(cfg=SPPEConfig(
        input_size=(256, 192),  # (H,W)
        device="cuda",          # or "cpu"
        sppe_cfg_yaml=Path("/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/AlphaPose/configs/coco/hrnet/256x192_w32_lr1e-3.yaml"),
        sppe_checkpoint=Path("/home/athena/DaRA_Thesis/AlphaPose_OSNet_Pipeline/AlphaPose/pretrained_models/pose_hrnet_w32_256x192.pth"),
        fp16=False
    ))


    # Detector
    y8 = None
    if args.detector.lower() == "yolov8":
        if not args.y8_model:
            log.error("--y8-model must be provided when using yolov8 detector.")
            raise SystemExit(2)

        y8 = YOLOv8Detector(
            model="yolov8s",              # or "yolov8m/l/x" depending on which you want
            weights=Path(args.y8_model),  # Path to your .pt file
            device=args.device,
            conf=args.y8_conf,
            iou=args.y8_iou,
            imgsz=args.y8_imgsz,
            half=args.y8_half,
        )
        log.info(
            "YOLOv8Detector ready: %s (device=%s, conf=%.2f, iou=%.2f, imgsz=%d, half=%s)",
            args.y8_model,
            args.device,
            args.y8_conf,
            args.y8_iou,
            args.y8_imgsz,
            args.y8_half,
        )


    # OSNet (ReID)
    osnet = OSNetExtractor(weight=str(Path(args.osnet_weights)), device=args.device)
    log.info("OSNetExtractor loaded: %s (device=%s)", args.osnet_weights, args.device)

    # Tracker (StrongSORT-style; your constructor)
    tracker = StrongSORTTracker(sim_thr=0.45, iou_thr=0.3, ttl=60)

    # Runner
    runner = InferenceRunner(
        alphapose=ap,
        osnet=osnet,
        tracker=tracker,
        detector=args.detector,
        y8=y8,
        draw=args.draw,
        save_embeds=args.save_embeds,
        log_every=args.log_every,
    )

    # ------------------ Open outputs using your io_utils ------------------
    vw = None
    jf = None
    try:
        out_video_path = None
        out_jsonl_path = None
        #cam_id = args.cam_id if hasattr(args, "cam_id") and args.cam_id else Path(args.video).stem

        if args.save_video:
            out_video_path = Path(args.out) / f"{Path(args.video).stem}_annot.mp4"
        if args.save_json:
            out_jsonl_path = Path(args.out) / f"{Path(args.video).stem}_poses.jsonl"

        # ------------------ Run ------------------
        log.info("----- RUN 1/1 | cam_id=%s | video=%s -----", base, str(video_path))
        runner.run_on_video(
            video_path=args.video,
            out_dir=out_dir,                   
            cam_id=video_path.stem,
            out_video_path=out_video_path,
            out_jsonl_path=out_jsonl_path,
            max_frames=args.max_frames,
            save_video=args.save_video,
            save_json=args.save_json,
        )
    finally:
        # If we created writers here, close them
        try:
            if vw is not None:
                vw.release()
        except Exception:
            pass
        try:
            if jf is not None:
                jf.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
