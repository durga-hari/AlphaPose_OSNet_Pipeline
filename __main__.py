#!/usr/bin/env python3
import argparse, logging, os
from AlphaPose_OSNet_Pipeline.Pipeline.pipeline import DaRAPipeline
from AlphaPose_OSNet_Pipeline.config import Config

def main():
    p = argparse.ArgumentParser("DaRA runner")

    # New-style flags (session-based)
    p.add_argument("--input-dir", default=None, help="Session folder containing videos")
    p.add_argument("--out", default=None, help="Output directory (overrides default)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--frame-skip", type=int, default=None)
    p.add_argument("--det-scale", type=float, default=None)
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--det-nms", type=float, default=None)
    p.add_argument("--no-draw", action="store_true", help="Disable drawing outputs")
    p.add_argument("--debug-dets", action="store_true")

    # Backward-compat flags (single video mode)
    p.add_argument("--video", default=None, help="Path to a single video file")
    p.add_argument("--cam-id", default=None, help="Camera ID label for single video")
    p.add_argument("--draw", action="store_true", help="(legacy) Enable drawing outputs")

    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("DaRA")

    # ---- Build base config from config.py (auto-fills model paths) ----
    cfg = Config()

    # ---- Apply overrides from CLI ----
    # Output & runtime
    if args.out:
        cfg.output_dir = args.out
    if args.device:
        cfg.device = args.device
    if args.frame_skip is not None:
        cfg.frame_skip = args.frame_skip
    if args.det_scale is not None:
        cfg.detector_scale = args.det_scale
    if args.det_conf is not None:
        cfg.det_conf_threshold = args.det_conf
    if args.det_nms is not None:
        cfg.det_nms_threshold = args.det_nms
    if args.no_draw:
        cfg.draw_outputs = False
    if args.draw:  # legacy flag wins if provided
        cfg.draw_outputs = True
    if args.debug_dets:
        cfg.debug_draw_dets = True

    # Session vs single video
    session_tag = os.path.basename(cfg.input_dir.rstrip("/"))
    if args.video:
        # Single video mode (legacy)
        if not os.path.isfile(args.video):
            raise SystemExit(f"Video not found: {args.video}")
        cfg.input_video_paths = [args.video]
        cfg.camera_ids = [args.cam_id or os.path.splitext(os.path.basename(args.video))[0]]
        session_tag = cfg.camera_ids[0]
        # If user supplied input-dir, ignore its discovered list; we’ve overridden paths above
    else:
        # Session mode (new) — allow overriding input_dir before refreshing derived lists
        if args.input_dir:
            cfg.input_dir = args.input_dir
        # Re-run post_init to refresh discovered videos & derived model paths if input_dir changed
        cfg.__post_init__()
        session_tag = os.path.basename(cfg.input_dir.rstrip("/"))

    # Validate files & inputs (weights, cfgs, videos, etc.)
    missing = cfg.validate()
    if missing:
        for m in missing:
            logger.error(m)
        raise SystemExit("Config validation failed. See errors above.")

    DaRAPipeline(cfg, logger, session_tag=session_tag).run()

if __name__ == "__main__":
    main()
