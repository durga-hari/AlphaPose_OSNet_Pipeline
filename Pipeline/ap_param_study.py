#!/usr/bin/env python3
"""
ap_param_study.py

Run identity-focused parameter studies for AlphaPose pipelines by REUSING your working logic from ap_pipeline.py.

Supports stacks:
  - deepsort_resnet   = DeepSORT + ResNet ReID
  - strongsort_osnet  = StrongSORT + OSNet ReID
  - both              = run both stacks

KEY CHANGE (what you asked):
  You can now trigger ALL experiments from ONE command:
    --run-all

It will execute a predefined suite (per variant) in a sensible order:
  Global stitching sweeps (cos_thr -> intra_merge_thr -> alpha -> time_slack)
  then tracker sweeps (DeepSORT or StrongSORT)

Outputs:
  <out_dir>/study/<variant>/<experiment>/
    - study_summary.csv
    - plots/... (updated incrementally per run)

Notes:
  - Window offsets need --make-clips (ffmpeg). Without it, only max_frames is respected.
  - This script does NOT sweep detector/pose thresholds.

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt

# ---- Import your working pipeline components from ap_pipeline.py ----
from ap_pipeline import (  # type: ignore
    SequentialPipeline,
    AlphaPoseCfg,
    merge_intra_camera_tracks,
    run_crosscam_union_stitch,
    inject_global_ids,
    merge_tracks_ap,
    count_global_individuals,
)


# --------------------------
# Utilities
# --------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def parse_windows(s: str) -> List[Tuple[int, int]]:
    """
    Parse "start:end,start:end,..." into list of (start_frame, length)
    Example: "0:5000,30000:35000" -> [(0,5000), (30000,5000)]
    """
    out: List[Tuple[int, int]] = []
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        a, b = p.split(":")
        start = int(a)
        end = int(b)
        if end <= start:
            raise ValueError(f"Bad window '{p}' (end must be > start).")
        out.append((start, end - start))
    return out

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    logging.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffmpeg_exists() -> bool:
    try:
        run_cmd(["ffmpeg", "-version"], check=True)
        return True
    except Exception:
        return False

def list_session_videos(video_path: Path) -> List[Path]:
    if video_path.is_dir():
        return sorted(video_path.glob("*.mp4"))
    return [video_path]

def make_clip_ffmpeg(
    src_video: Path,
    dst_video: Path,
    start_frame: int,
    length: int,
    fps_hint: Optional[float] = None,
) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Install ffmpeg or run without --make-clips.")

    fps = fps_hint
    if fps is None:
        try:
            pr = run_cmd([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(src_video)
            ], check=True)
            rate = pr.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(rate)
        except Exception:
            fps = 25.0

    start_sec = start_frame / float(fps)
    dur_sec = length / float(fps)

    dst_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-ss", f"{start_sec:.6f}",
        "-t", f"{dur_sec:.6f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst_video),
    ]
    pr = run_cmd(cmd, check=True)
    if pr.stderr:
        logging.debug("ffmpeg stderr (trim): %s", pr.stderr[-1200:])

def prepare_window_videos(
    session_dir: Path,
    window: Tuple[int, int],
    work_dir: Path,
    make_clips: bool,
) -> Path:
    start, length = window
    if not make_clips:
        return session_dir

    clips_dir = work_dir / f"clips_s{start}_n{length}"
    if clips_dir.exists():
        return clips_dir

    clips_dir.mkdir(parents=True, exist_ok=True)
    vids = list_session_videos(session_dir)
    if not vids:
        raise RuntimeError(f"No videos found in {session_dir}")

    logging.info("Creating window clips: start=%d len=%d into %s", start, length, clips_dir)
    for v in vids:
        dst = clips_dir / v.name
        make_clip_ffmpeg(v, dst, start_frame=start, length=length)
    return clips_dir

def count_local_ids_in_jsonl(jsonl_path: Optional[Path]) -> int:
    if not jsonl_path or not jsonl_path.exists():
        return 0
    seen = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for p in rec.get("poses", []):
                tid = p.get("id", None)
                if tid is not None:
                    try:
                        seen.add(int(tid))
                    except Exception:
                        pass
    return len(seen)

def component_sizes_from_union_map(union_map: Dict[str, Dict[int, int]]) -> Dict[int, int]:
    sizes: Dict[int, int] = {}
    for _, m in union_map.items():
        for _, gid in (m or {}).items():
            gid_int = int(gid)
            sizes[gid_int] = sizes.get(gid_int, 0) + 1
    return sizes

def safe_json_load(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

def summarize_local_tracks_total(per_cam_local_ids_json: str) -> int:
    d = safe_json_load(per_cam_local_ids_json)
    if not isinstance(d, dict):
        return 0
    total = 0
    for _, v in d.items():
        try:
            total += int(v)
        except Exception:
            pass
    return total


# --------------------------
# Plotting
# --------------------------

def get_sweep_x_and_label(experiment: str, row: Dict[str, Any]) -> Tuple[Optional[float], str]:
    exp = experiment.lower().strip()
    tracker_params = safe_json_load(row.get("tracker_params_json", ""))

    if exp == "stitch_cos_thr":
        return float(row["stitch_cos_thr"]), "stitch_cos_thr"
    if exp == "stitch_alpha":
        return float(row["stitch_alpha"]), "stitch_alpha"
    if exp == "stitch_time_slack":
        return float(row["stitch_time_slack"]), "stitch_time_slack"
    if exp == "intra_merge_thr":
        return float(row["intra_merge_thr"]), "intra_merge_thr"

    if exp == "tracker_ds_max_age":
        v = tracker_params.get("max_age", None)
        return (float(v) if v is not None else None), "deepsort.max_age"
    if exp == "tracker_ds_max_cosine":
        v = tracker_params.get("max_cosine_distance", None)
        return (float(v) if v is not None else None), "deepsort.max_cosine_distance"
    if exp == "tracker_ds_n_init":
        v = tracker_params.get("n_init", None)
        return (float(v) if v is not None else None), "deepsort.n_init"
    if exp == "tracker_ds_max_iou":
        v = tracker_params.get("max_iou_distance", None)
        return (float(v) if v is not None else None), "deepsort.max_iou_distance"

    if exp == "tracker_ss_sim_thr":
        v = tracker_params.get("sim_thr", None)
        return (float(v) if v is not None else None), "strongsort.sim_thr"
    if exp == "tracker_ss_ttl":
        v = tracker_params.get("ttl", None)
        return (float(v) if v is not None else None), "strongsort.ttl"
    if exp == "tracker_ss_iou_thr":
        v = tracker_params.get("iou_thr", None)
        return (float(v) if v is not None else None), "strongsort.iou_thr"

    return None, "parameter"

def save_plots_for_window_variant(
    rows: List[Dict[str, Any]],
    experiment: str,
    out_dir: Path,
    title_prefix: str,
) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    xs: List[float] = []
    gids: List[int] = []
    lcs: List[int] = []
    ltot: List[int] = []
    x_label = "parameter"

    for r in rows:
        x, lbl = get_sweep_x_and_label(experiment, r)
        if x is None:
            continue
        x_label = lbl
        xs.append(float(x))
        gids.append(int(float(r["global_ids"])))
        lcs.append(int(float(r["largest_component_local_tracks"])))
        ltot.append(summarize_local_tracks_total(r.get("per_cam_local_ids", "")))

    if not xs:
        return

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    gids = [gids[i] for i in order]
    lcs = [lcs[i] for i in order]
    ltot = [ltot[i] for i in order]

    plt.figure()
    plt.plot(xs, gids, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("global_ids")
    plt.title(f"{title_prefix} | global_ids vs {x_label}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "global_ids_vs_param.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, lcs, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("largest_component_local_tracks")
    plt.title(f"{title_prefix} | largest_component vs {x_label}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "largest_component_vs_param.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, ltot, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("total_local_tracks (sum over cameras)")
    plt.title(f"{title_prefix} | local_tracks_total vs {x_label}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "local_tracks_total_vs_param.png", dpi=200)
    plt.close()


# --------------------------
# Variant handling
# --------------------------

def apply_variant(
    cfg: dict,
    variant: str,
    osnet_weights: Optional[str],
    resnet_weights: Optional[str],
) -> dict:
    v = variant.lower().strip()
    cfg = deepcopy(cfg)

    cfg.setdefault("reid", {})
    cfg.setdefault("tracker", {})

    if v == "deepsort_resnet":
        cfg["tracker"]["type"] = "deepsort"
        cfg["reid"]["type"] = "resnet"
        if resnet_weights:
            cfg["reid"]["weights"] = resnet_weights
    elif v == "strongsort_osnet":
        cfg["tracker"]["type"] = "strongsort"
        cfg["reid"]["type"] = "osnet"
        if osnet_weights:
            cfg["reid"]["weights"] = osnet_weights
    else:
        raise ValueError(f"Unknown variant '{variant}'")

    w = cfg.get("reid", {}).get("weights", None)
    if w and not Path(w).exists():
        logging.warning("ReID weights path does not exist: %s", w)

    return cfg


# --------------------------
# Pipeline builder (matches ap_pipeline.py main)
# --------------------------

def build_sequential_pipeline(cfg: dict) -> SequentialPipeline:
    detector_cfg = cfg["detector"]
    pose_cfg = cfg["pose"]
    reid_cfg = cfg.get("reid", {})
    tracker_cfg = cfg.get("tracker", {})
    pipe_cfg = cfg.get("pipeline", {})

    detector_args = dict(
        model=detector_cfg["model"],
        weights=detector_cfg["weights"],
        device=detector_cfg.get("device", "cuda"),
        conf=float(detector_cfg.get("conf", 0.5)),
        iou=float(detector_cfg.get("iou", 0.6)),
        imgsz=int(detector_cfg.get("imgsz", 640)),
        half=bool(detector_cfg.get("half", False)),
    )

    pose_args = dict(
        config=AlphaPoseCfg(
            input_size=tuple(pose_cfg.get("input_size", [256, 192])),
            device=pose_cfg.get("device", "cuda"),
            cfg_yaml=pose_cfg["cfg"],
            checkpoint=pose_cfg["ckpt"],
            fp16=pose_cfg.get("fp16", False),
        )
    )

    reid_type = (reid_cfg.get("type", "none") or "none").lower()
    reid_args = (
        dict(weights=reid_cfg["weights"], device=reid_cfg.get("device", "cuda"))
        if reid_type != "none"
        else {}
    )

    tracker_type = (tracker_cfg.get("type", "none") or "none").lower()
    if tracker_type == "deepsort":
        tracker_args = dict(
            max_cosine_distance=float(tracker_cfg.get("max_cosine_distance", 0.2)),
            max_iou_distance=float(tracker_cfg.get("max_iou_distance", 0.7)),
            max_age=int(tracker_cfg.get("max_age", 30)),
            n_init=int(tracker_cfg.get("n_init", 3)),
        )
    elif tracker_type == "strongsort":
        tracker_args = dict(
            sim_thr=float(tracker_cfg.get("sim_thr", 0.55)),
            iou_thr=float(tracker_cfg.get("iou_thr", 0.6)),
            ttl=int(tracker_cfg.get("ttl", 80)),
        )
    else:
        tracker_args = {}

    return SequentialPipeline(
        detector_name=detector_cfg["type"],
        detector_args=detector_args,
        pose_name=pose_cfg["type"],
        pose_args=pose_args,
        reid_name=reid_type,
        reid_args=reid_args,
        tracker_name=tracker_type,
        tracker_args=tracker_args,
        draw=bool(pipe_cfg.get("draw", False)),
        kpt_thresh=float(pipe_cfg.get("kpt_thresh", 0.2)),
        dataset_hint=str(pipe_cfg.get("kp_set", "coco_wholebody")).lower(),
        debug_heatmaps=bool(pipe_cfg.get("debug_heatmaps", False)),
    )


# --------------------------
# Experiment specs
# --------------------------

@dataclass(frozen=True)
class RunSpec:
    tag: str
    cfg_overrides: Dict[str, Any]
    intra_merge_thr: float
    stitch_cos_thr: float
    stitch_alpha: float
    stitch_time_slack: int


def get_experiment_runs(experiment: str) -> List[RunSpec]:
    exp = experiment.lower().strip()

    default_intra = 0.75
    default_cos = 0.55
    default_alpha = 0.60
    default_slack = 30

    if exp == "tracker_ds_max_age":
        ages = [15, 30, 60, 120]
        return [
            RunSpec(
                tag=f"ds_max_age_{a}",
                cfg_overrides={"tracker": {"type": "deepsort", "max_age": a}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for a in ages
        ]

    if exp == "tracker_ds_max_cosine":
        vals = [0.10, 0.20, 0.30, 0.40]
        return [
            RunSpec(
                tag=f"ds_max_cos_{v:.2f}",
                cfg_overrides={"tracker": {"type": "deepsort", "max_cosine_distance": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "tracker_ds_n_init":
        vals = [1, 3, 5]
        return [
            RunSpec(
                tag=f"ds_n_init_{v}",
                cfg_overrides={"tracker": {"type": "deepsort", "n_init": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "tracker_ds_max_iou":
        vals = [0.5, 0.7, 0.9]
        return [
            RunSpec(
                tag=f"ds_max_iou_{v:.2f}",
                cfg_overrides={"tracker": {"type": "deepsort", "max_iou_distance": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "tracker_ss_sim_thr":
        vals = [0.35, 0.45, 0.55, 0.65]
        return [
            RunSpec(
                tag=f"ss_sim_thr_{v:.2f}",
                cfg_overrides={"tracker": {"type": "strongsort", "sim_thr": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "tracker_ss_ttl":
        vals = [50, 100, 150, 250]
        return [
            RunSpec(
                tag=f"ss_ttl_{v}",
                cfg_overrides={"tracker": {"type": "strongsort", "ttl": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "tracker_ss_iou_thr":
        vals = [0.3, 0.5, 0.6, 0.7]
        return [
            RunSpec(
                tag=f"ss_iou_thr_{v:.2f}",
                cfg_overrides={"tracker": {"type": "strongsort", "iou_thr": v}},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for v in vals
        ]

    if exp == "stitch_cos_thr":
        thr = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        return [
            RunSpec(
                tag=f"st_cos_thr_{t:.2f}",
                cfg_overrides={},
                intra_merge_thr=default_intra,
                stitch_cos_thr=t,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for t in thr
        ]

    if exp == "stitch_alpha":
        alphas = [1.0, 0.8, 0.6, 0.4]
        return [
            RunSpec(
                tag=f"st_alpha_{a:.2f}",
                cfg_overrides={},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=a,
                stitch_time_slack=default_slack,
            )
            for a in alphas
        ]

    if exp == "stitch_time_slack":
        vals = [0, 30, 150, 300]
        return [
            RunSpec(
                tag=f"st_time_slack_{v}",
                cfg_overrides={},
                intra_merge_thr=default_intra,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=v,
            )
            for v in vals
        ]

    if exp == "intra_merge_thr":
        thrs = [0.65, 0.75, 0.85]
        return [
            RunSpec(
                tag=f"intra_thr_{t:.2f}",
                cfg_overrides={},
                intra_merge_thr=t,
                stitch_cos_thr=default_cos,
                stitch_alpha=default_alpha,
                stitch_time_slack=default_slack,
            )
            for t in thrs
        ]

    raise ValueError(f"Unknown experiment '{experiment}'")


def is_run_compatible_with_variant(run: RunSpec, variant: str) -> bool:
    v = variant.lower().strip()
    trk = (run.cfg_overrides.get("tracker", {}).get("type") or "").lower()
    if not trk:
        return True
    if v == "deepsort_resnet":
        return trk == "deepsort"
    if v == "strongsort_osnet":
        return trk == "strongsort"
    return True


# --------------------------
# Core run
# --------------------------

def run_one(
    base_cfg: dict,
    run: RunSpec,
    window_dir: Path,
    out_root: Path,
    max_frames_override: Optional[int],
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    deep_update(cfg, run.cfg_overrides)

    cfg["out_dir"] = str(out_root)

    if max_frames_override is not None:
        cfg.setdefault("pipeline", {})["max_frames"] = int(max_frames_override)

    dump_yaml(cfg, out_root / "run_config.yaml")

    pipe = build_sequential_pipeline(cfg)

    vids = sorted(window_dir.glob("*.mp4")) if window_dir.is_dir() else [window_dir]
    if not vids:
        raise RuntimeError(f"No videos to process in {window_dir}")

    max_frames_cfg = int(cfg.get("pipeline", {}).get("max_frames", 0) or 0)
    max_frames = None if max_frames_cfg <= 0 else max_frames_cfg

    for v in vids:
        cam_out = out_root / v.stem
        cam_out.mkdir(parents=True, exist_ok=True)
        logging.info("[Run %s] Processing %s", run.tag, v.name)
        pipe.run(
            video_path=v,
            out_dir=cam_out,
            save_video=bool(cfg.get("pipeline", {}).get("save_video", False)),
            save_json=bool(cfg.get("pipeline", {}).get("save_json", True)),
            save_embeds=bool(cfg.get("pipeline", {}).get("save_embeds", True)),
            max_frames=max_frames,
        )

    merge_intra_camera_tracks(out_root, sim_thr=float(run.intra_merge_thr))

    union_map = run_crosscam_union_stitch(
        out_root,
        cos_thr=float(run.stitch_cos_thr),
        alpha=float(run.stitch_alpha),
        time_slack=int(run.stitch_time_slack),
    )

    inject_global_ids(out_root, union_map, suffix="_poses_union.jsonl")
    merged_path = out_root / "merged_global_tracks_union.jsonl"
    merge_tracks_ap(out_root, union_map, merged_path)

    gid_count = count_global_individuals(merged_path) if merged_path.exists() else 0
    comp_sizes = component_sizes_from_union_map(union_map)
    largest_comp = max(comp_sizes.values()) if comp_sizes else 0

    per_cam_local_ids: Dict[str, int] = {}
    for cam_dir in sorted([d for d in out_root.iterdir() if d.is_dir()]):
        poses_jsonl = next(cam_dir.glob("*_poses.jsonl"), None)
        per_cam_local_ids[cam_dir.name] = count_local_ids_in_jsonl(poses_jsonl) if poses_jsonl else 0

    tracker_type = (cfg.get("tracker", {}).get("type") or "none").lower()
    reid_type = (cfg.get("reid", {}).get("type") or "none").lower()

    return {
        "tag": run.tag,
        "tracker": tracker_type,
        "tracker_params_json": json.dumps(cfg.get("tracker", {}), sort_keys=True),
        "reid": reid_type,
        "reid_weights": str(cfg.get("reid", {}).get("weights", "")),
        "intra_merge_thr": run.intra_merge_thr,
        "stitch_cos_thr": run.stitch_cos_thr,
        "stitch_alpha": run.stitch_alpha,
        "stitch_time_slack": run.stitch_time_slack,
        "global_ids": gid_count,
        "largest_component_local_tracks": largest_comp,
        "per_cam_local_ids": json.dumps(per_cam_local_ids, sort_keys=True),
        "out_dir": str(out_root),
    }


# --------------------------
# Experiment suite runner
# --------------------------

def suite_for_variant(variant: str) -> List[str]:
    v = variant.lower().strip()
    common = ["stitch_cos_thr", "intra_merge_thr", "stitch_alpha", "stitch_time_slack"]
    if v == "deepsort_resnet":
        return common + ["tracker_ds_max_cosine", "tracker_ds_max_age", "tracker_ds_n_init", "tracker_ds_max_iou"]
    if v == "strongsort_osnet":
        return common + ["tracker_ss_sim_thr", "tracker_ss_ttl", "tracker_ss_iou_thr"]
    raise ValueError(f"Unknown variant '{variant}'")


def run_experiment(
    *,
    base_cfg_raw: dict,
    session_dir: Path,
    base_out: Path,
    variant: str,
    experiment: str,
    windows: List[Tuple[int, int]],
    make_clips: bool,
    max_frames_arg: int,
    study_out_override: Optional[Path],
    clean: bool,
    osnet_weights: Optional[str],
    resnet_weights: Optional[str],
) -> None:
    base_cfg_variant = apply_variant(
        base_cfg_raw,
        variant=variant,
        osnet_weights=osnet_weights,
        resnet_weights=resnet_weights,
    )

    if study_out_override:
        study_out = study_out_override / variant / experiment
    else:
        study_out = base_out / "study" / variant / experiment

    if clean and study_out.exists():
        shutil.rmtree(study_out)
    study_out.mkdir(parents=True, exist_ok=True)

    runs = get_experiment_runs(experiment)
    all_rows: List[Dict[str, Any]] = []

    summary_path = study_out / "study_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "variant",
                "experiment",
                "window_start", "window_len",
                "tag",
                "tracker", "tracker_params_json",
                "reid", "reid_weights",
                "intra_merge_thr",
                "stitch_cos_thr", "stitch_alpha", "stitch_time_slack",
                "global_ids", "largest_component_local_tracks",
                "per_cam_local_ids",
                "out_dir",
            ],
        )
        writer.writeheader()

        logging.info("Running: variant=%s experiment=%s runs=%d windows=%d make_clips=%s",
                     variant, experiment, len(runs), len(windows), make_clips)

        for (w_start, w_len) in windows:
            window_work = study_out / f"window_s{w_start}_n{w_len}"
            window_work.mkdir(parents=True, exist_ok=True)

            window_dir = prepare_window_videos(
                session_dir=session_dir,
                window=(w_start, w_len),
                work_dir=window_work,
                make_clips=make_clips,
            )

            # max_frames behavior:
            # - if user gave --max-frames, use it
            # - else if make_clips, set to window length
            # - else leave as config default
            max_frames_override: Optional[int] = None
            if max_frames_arg > 0:
                max_frames_override = max_frames_arg
            elif make_clips:
                max_frames_override = w_len

            for run in runs:
                if not is_run_compatible_with_variant(run, variant):
                    continue

                run_out = window_work / "runs" / run.tag
                if run_out.exists():
                    logging.info("Skipping existing run output: %s", run_out)
                    continue
                run_out.mkdir(parents=True, exist_ok=True)

                cfg_run = deepcopy(base_cfg_variant)
                cfg_run["video_path"] = str(window_dir)
                cfg_run["out_dir"] = str(run_out)

                t0 = time.time()
                metrics = run_one(
                    base_cfg=cfg_run,
                    run=run,
                    window_dir=Path(cfg_run["video_path"]),
                    out_root=run_out,
                    max_frames_override=max_frames_override,
                )
                dt = time.time() - t0

                row = {
                    "variant": variant,
                    "experiment": experiment,
                    "window_start": w_start,
                    "window_len": w_len,
                    "tag": metrics["tag"],
                    "tracker": metrics["tracker"],
                    "tracker_params_json": metrics["tracker_params_json"],
                    "reid": metrics["reid"],
                    "reid_weights": metrics["reid_weights"],
                    "intra_merge_thr": metrics["intra_merge_thr"],
                    "stitch_cos_thr": metrics["stitch_cos_thr"],
                    "stitch_alpha": metrics["stitch_alpha"],
                    "stitch_time_slack": metrics["stitch_time_slack"],
                    "global_ids": metrics["global_ids"],
                    "largest_component_local_tracks": metrics["largest_component_local_tracks"],
                    "per_cam_local_ids": metrics["per_cam_local_ids"],
                    "out_dir": metrics["out_dir"],
                }

                writer.writerow(row)
                all_rows.append(row)

                logging.info(
                    "[DONE] %s | %s | window %d:%d | global_ids=%s | largest_comp=%s | %.1fs",
                    variant, run.tag, w_start, w_start + w_len,
                    metrics["global_ids"], metrics["largest_component_local_tracks"], dt
                )

                plots_root = study_out / "plots" / f"window_s{w_start}_n{w_len}"
                title = f"{variant} | {experiment} | window {w_start}:{w_start + w_len}"
                rows_here = [
                    r for r in all_rows
                    if r["variant"] == variant
                    and r["experiment"] == experiment
                    and int(r["window_start"]) == w_start
                    and int(r["window_len"]) == w_len
                ]
                save_plots_for_window_variant(
                    rows=rows_here,
                    experiment=experiment,
                    out_dir=plots_root,
                    title_prefix=title,
                )

    logging.info("Wrote: %s", summary_path)
    logging.info("Plots: %s", (study_out / "plots"))


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to ap_pipeline_config.yaml")

    ap.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["deepsort_resnet", "strongsort_osnet", "both"],
        help="Which combination to test."
    )

    ap.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Single experiment to run (ignored if --run-all)."
    )

    ap.add_argument(
        "--run-all",
        action="store_true",
        help="Run the full experiment suite for the selected variant(s) in one invocation."
    )

    ap.add_argument(
        "--windows",
        type=str,
        default="",
        help='Optional windows "start:end,start:end,...". Example: "0:5000,30000:35000"',
    )

    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Override pipeline.max_frames. 0 = use config (or window length if clipping).",
    )

    ap.add_argument(
        "--make-clips",
        action="store_true",
        help="Use ffmpeg to create windowed clips per camera (recommended for true start offsets).",
    )

    ap.add_argument(
        "--study-out",
        type=str,
        default="",
        help="Optional output root. If set, outputs to <study-out>/<variant>/<experiment>/",
    )

    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output folders for experiments before running.",
    )

    ap.add_argument(
        "--osnet-weights",
        type=str,
        default="",
        help="Override OSNet weights path (used by strongsort_osnet).",
    )

    ap.add_argument(
        "--resnet-weights",
        type=str,
        default="",
        help="Override ResNet weights path (used by deepsort_resnet).",
    )

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg_path = Path(args.config).resolve()
    base_cfg_raw = load_yaml(cfg_path)

    session_dir = Path(base_cfg_raw["video_path"]).resolve()
    if not session_dir.exists():
        raise SystemExit(f"video_path not found: {session_dir}")

    base_out = Path(base_cfg_raw["out_dir"]).resolve()
    study_out_override = Path(args.study_out).resolve() if args.study_out.strip() else None

    windows = parse_windows(args.windows)
    if not windows:
        mf = args.max_frames if args.max_frames > 0 else int(base_cfg_raw.get("pipeline", {}).get("max_frames", 0) or 0)
        if mf <= 0:
            mf = 5000
        windows = [(0, mf)]
        logging.info("No --windows provided. Using single window length=%d (no start offset).", mf)

    osnet_w = args.osnet_weights.strip() or None
    resnet_w = args.resnet_weights.strip() or None

    variants = ["deepsort_resnet", "strongsort_osnet"] if args.variant == "both" else [args.variant]

    if args.run_all:
        for v in variants:
            for exp in suite_for_variant(v):
                run_experiment(
                    base_cfg_raw=base_cfg_raw,
                    session_dir=session_dir,
                    base_out=base_out,
                    variant=v,
                    experiment=exp,
                    windows=windows if exp.startswith("stitch_") or exp == "intra_merge_thr" else [(windows[0][0], windows[0][1])],
                    make_clips=args.make_clips,
                    max_frames_arg=args.max_frames,
                    study_out_override=study_out_override,
                    clean=args.clean,
                    osnet_weights=osnet_w,
                    resnet_weights=resnet_w,
                )
        return

    # Single experiment mode
    if not args.experiment.strip():
        raise SystemExit("Provide --experiment <name> OR use --run-all")

    for v in variants:
        run_experiment(
            base_cfg_raw=base_cfg_raw,
            session_dir=session_dir,
            base_out=base_out,
            variant=v,
            experiment=args.experiment.strip(),
            windows=windows,
            make_clips=args.make_clips,
            max_frames_arg=args.max_frames,
            study_out_override=study_out_override,
            clean=args.clean,
            osnet_weights=osnet_w,
            resnet_weights=resnet_w,
        )


if __name__ == "__main__":
    main()
