#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of unsupervised metrics for a single DaRA AP pipeline configuration.

Reads (from --ap-root):
  - per_video.csv
  - per_track.csv
  - limb_lengths.csv
  - jitter_timeseries.csv
  - per_frame_metrics.csv
Optionally (from --global-file under --ap-root):
  - merged_global_tracks_union.jsonl (or similar)

Produces (under --out-dir inside --ap-root):
  Global summary:
    - detector_coverage_vs_persistence.png
    - limb_length_cv_per_video.png
    - jitter_boxplot_by_cam.png
    - track_length_hist.png          (zoomed, percentile-clipped; default P95)
    - track_length_hist_full.png     (full range, for debugging)

  Per-camera time series:
    - detector_tracking_timeseries_cam_<CAM>.png
      (n_dets, mean_det_score, n_tracks_active vs time_bin)
    - jitter_timeseries_cam_<CAM>.png
      (mean jitter vs time_bin)
    - limb_lengths_timeseries_cam_<CAM>.png
      (per-bone limb length vs time_bin as subplots)

  Global ID stitching:
    - globalid_camera_overlap_heatmap.png
    - globalid_camera_count_hist.png
"""

from __future__ import annotations

import argparse
from itertools import combinations
from math import ceil
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def load_tables(root: Path) -> dict:
    tables = {}
    tables["per_video"] = pd.read_csv(root / "per_video.csv")
    tables["per_track"] = pd.read_csv(root / "per_track.csv")
    tables["limb"] = pd.read_csv(root / "limb_lengths.csv")
    tables["jitter"] = pd.read_csv(root / "jitter_timeseries.csv")
    tables["per_frame"] = pd.read_csv(root / "per_frame_metrics.csv")
    return tables


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------
# Global summary plots
# ---------------------------------------------------------------------

def plot_detector_coverage_vs_persistence(
    per_video: pd.DataFrame,
    config_label: str,
    out_dir: Path,
):
    if "mean_dets_per_frame" not in per_video.columns or \
       "temporal_persistence_rate" not in per_video.columns:
        print("[WARN] detector metrics not found in per_video.csv")
        return

    x = per_video["mean_dets_per_frame"]
    y = per_video["temporal_persistence_rate"]

    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    for _, row in per_video.iterrows():
        try:
            label = f"{row.get('cam_id', '')}:{Path(str(row.get('file', ''))).stem}"
        except Exception:
            label = str(row.get("cam_id", ""))
        if pd.notna(row["mean_dets_per_frame"]) and pd.notna(row["temporal_persistence_rate"]):
            plt.annotate(
                label,
                (row["mean_dets_per_frame"], row["temporal_persistence_rate"]),
                fontsize=7,
                alpha=0.7,
            )

    plt.xlabel("Mean detections per frame (coverage)")
    plt.ylabel("Temporal persistence rate")
    plt.title(f"Detector coverage vs persistence ({config_label})")

    out_path = out_dir / "detector_coverage_vs_persistence.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_limb_length_cv_per_video(
    per_video: pd.DataFrame,
    config_label: str,
    out_dir: Path,
):
    col = "mean_limb_length_cv_over_tracks"
    if col not in per_video.columns:
        print("[WARN] limb-length CV metric not found in per_video.csv")
        return

    vals = per_video[col].dropna()
    if vals.empty:
        print("[WARN] no non-NaN limb-length CV values in per_video.csv")
        return

    plt.figure()
    plt.boxplot([vals.values], labels=[config_label], showfliers=False)
    plt.ylabel("Mean limb length CV over tracks")
    plt.title("Pose stability: limb length consistency")

    out_path = out_dir / "limb_length_cv_per_video.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_jitter_boxplot_by_cam(
    per_video: pd.DataFrame,
    config_label: str,
    out_dir: Path,
):
    if "mean_pose_jitter" not in per_video.columns or "cam_id" not in per_video.columns:
        print("[WARN] pose jitter metrics or cam_id missing in per_video.csv")
        return

    cams = per_video["cam_id"].unique()
    data = []
    labels = []

    for cam in cams:
        sub = per_video[per_video["cam_id"] == cam]
        vals = sub["mean_pose_jitter"].dropna()
        if vals.empty:
            continue
        data.append(vals.values)
        labels.append(str(cam))

    if not data:
        print("[WARN] no jitter data per camera to plot (boxplot)")
        return

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Mean pose jitter")
    plt.xlabel("Camera")
    plt.title(f"Pose jitter by camera ({config_label})")

    out_path = out_dir / "jitter_boxplot_by_cam.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_track_length_hist(
    per_track: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    clip_percentile: float = 95.0,
    log_y: bool = False,
):
    """
    Plot track length distributions in a way that is robust to extreme outliers.

    - track_length_hist.png      : zoomed to the given percentile (default 95th)
    - track_length_hist_full.png : full range, mainly for debugging
    """
    if "track_len" not in per_track.columns:
        print("[WARN] track_len not found in per_track.csv")
        return

    vals = per_track["track_len"].dropna().to_numpy()
    if vals.size == 0:
        print("[WARN] no track_len values in per_track.csv")
        return

    # Full-range histogram (debug / sanity)
    plt.figure()
    bins_full = np.linspace(0, vals.max(), 40)
    plt.hist(vals, bins=bins_full, alpha=0.7)
    plt.xlabel("Track length (frames)")
    plt.ylabel("Count")
    if log_y:
        plt.yscale("log")
    plt.title(f"Track length distribution – full range ({config_label})")

    out_path_full = out_dir / "track_length_hist_full.png"
    plt.savefig(out_path_full, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path_full}")

    # Percentile-clipped histogram (main view)
    if clip_percentile <= 0 or clip_percentile > 100:
        clip_percentile = 95.0

    clipped_max = np.percentile(vals, clip_percentile)
    if clipped_max <= 0:
        clipped_max = vals.max()

    vals_clipped = vals[vals <= clipped_max]
    if vals_clipped.size == 0:
        print("[WARN] no values below clip percentile; falling back to full range for zoomed hist")
        vals_clipped = vals
        clipped_max = vals.max()

    plt.figure()
    bins_zoom = np.linspace(0, clipped_max, 40)
    plt.hist(vals_clipped, bins=bins_zoom, alpha=0.7)
    plt.xlabel("Track length (frames)")
    plt.ylabel("Count")
    if log_y:
        plt.yscale("log")

    title_suffix = (
        f"≤ P{clip_percentile:.1f} (max={int(clipped_max)})"
        if clipped_max < vals.max()
        else "full range (no clipping)"
    )
    plt.title(f"Track length distribution ({config_label}, {title_suffix})")

    out_path_zoom = out_dir / "track_length_hist.png"
    plt.savefig(out_path_zoom, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path_zoom}")


# ---------------------------------------------------------------------
# Per-camera detector + tracking time series
# ---------------------------------------------------------------------

def plot_detector_tracking_timeseries_per_camera(
    per_frame: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    time_bin: int,
):
    required = {"cam_id", "frame_idx", "n_dets", "mean_det_score", "n_tracks_active"}
    if not required.issubset(per_frame.columns):
        print("[WARN] per_frame_metrics.csv missing required columns")
        return

    df = per_frame.copy()
    df["t_bin"] = (df["frame_idx"] // time_bin).astype(int)

    agg = (
        df.groupby(["cam_id", "t_bin"], as_index=False)[
            ["n_dets", "mean_det_score", "n_tracks_active"]
        ].mean()
    )

    for cam, sub in agg.groupby("cam_id"):
        sub = sub.sort_values("t_bin")
        t = sub["t_bin"] * time_bin

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(t, sub["n_dets"])
        axes[0].set_ylabel("Detections / bin")
        axes[0].set_title(f"Detections vs time – Camera {cam}")

        axes[1].plot(t, sub["mean_det_score"])
        axes[1].set_ylabel("Mean det score")

        axes[2].plot(t, sub["n_tracks_active"])
        axes[2].set_ylabel("Active tracks / bin")
        axes[2].set_xlabel("Frame index (binned)")

        fig.suptitle(f"Detector & tracking time series – Camera {cam} ({config_label})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = out_dir / f"detector_tracking_timeseries_cam_{cam}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


# ---------------------------------------------------------------------
# Per-camera jitter and limb-length time series
# ---------------------------------------------------------------------

def plot_jitter_timeseries_per_camera(
    jitter_df: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    time_bin: int,
):
    required = {"cam_id", "frame_idx", "jitter_value"}
    if not required.issubset(jitter_df.columns):
        print("[WARN] jitter_timeseries.csv missing required columns "
              "(cam_id, frame_idx, jitter_value)")
        return

    if jitter_df.empty:
        print("[WARN] jitter_timeseries.csv is empty")
        return

    df = jitter_df.copy()
    df["t_bin"] = (df["frame_idx"] // time_bin).astype(int)

    agg = (
        df.groupby(["cam_id", "t_bin"], as_index=False)["jitter_value"]
        .mean()
    )

    for cam, sub in agg.groupby("cam_id"):
        sub = sub.sort_values("t_bin")
        plt.figure()
        plt.plot(sub["t_bin"] * time_bin, sub["jitter_value"])
        plt.xlabel("Frame index (binned)")
        plt.ylabel("Mean jitter (pixels)")
        plt.title(f"Jitter over time – Camera {cam} ({config_label})")

        out_path = out_dir / f"jitter_timeseries_cam_{cam}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] wrote {out_path}")


def plot_limb_length_timeseries_per_camera(
    limb_df: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    time_bin: int,
    max_bones_per_cam: int,
):
    required = {"cam_id", "frame_idx", "bone", "limb_length"}
    if not required.issubset(limb_df.columns):
        print("[WARN] limb_lengths.csv missing required columns "
              "(cam_id, frame_idx, bone, limb_length)")
        return

    if limb_df.empty:
        print("[WARN] limb_lengths.csv is empty")
        return

    df = limb_df.copy()
    df["t_bin"] = (df["frame_idx"] // time_bin).astype(int)

    agg = (
        df.groupby(["cam_id", "bone", "t_bin"], as_index=False)["limb_length"]
        .mean()
    )

    for cam, sub_cam in agg.groupby("cam_id"):
        bones = sorted(sub_cam["bone"].unique())
        bones_sel = bones[:max_bones_per_cam]
        if not bones_sel:
            continue

        n_bones = len(bones_sel)
        n_cols = 3 if n_bones >= 3 else n_bones
        n_rows = ceil(n_bones / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=True,
        )
        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = [axes]

        for ax, bone in zip(axes, bones_sel):
            s = sub_cam[sub_cam["bone"] == bone].sort_values("t_bin")
            if s.empty:
                ax.set_visible(False)
                continue
            ax.plot(s["t_bin"] * time_bin, s["limb_length"])
            ax.set_title(str(bone), fontsize=8)
            ax.set_ylabel("Length (px)")

        for ax in axes[n_bones:]:
            ax.set_visible(False)

        fig.suptitle(f"Limb length over time – Camera {cam} ({config_label})")
        fig.supxlabel("Frame index (binned)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = out_dir / f"limb_lengths_timeseries_cam_{cam}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


# ---------------------------------------------------------------------
# Global ID visualization
# ---------------------------------------------------------------------

def build_global_tracks(global_path: Path):
    """
    Build global_id -> list of entries {cam_id, frame_idx}
    Supports both:
      A) flattened records with global_id, cam_id, frame/frame_idx
      B) AP-style: {cam_id, frame, poses: [{global_id, ...}, ...]}
    """
    tracks = {}
    for rec in load_jsonl(global_path):
        if "poses" in rec:
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            poses = rec.get("poses") or []
            for person in poses:
                gid = person.get("global_id") or person.get("gid")
                if gid is None:
                    continue
                tracks.setdefault(gid, []).append({
                    "cam_id": cam_id,
                    "frame_idx": frame_idx,
                })
        else:
            gid = rec.get("global_id") or rec.get("gid")
            if gid is None:
                continue
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            tracks.setdefault(gid, []).append({
                "cam_id": cam_id,
                "frame_idx": frame_idx,
            })
    return tracks


def plot_globalid_camera_overlap(global_path: Path, out_dir: Path):
    if not global_path.exists():
        print(f"[WARN] global file {global_path} not found – skipping global ID plots")
        return

    global_tracks = build_global_tracks(global_path)
    if not global_tracks:
        print("[WARN] global tracks empty – skipping global ID plots")
        return

    # camera set per global ID
    cam_sets = {}
    for gid, entries in global_tracks.items():
        cams = {e.get("cam_id") for e in entries if e.get("cam_id") is not None}
        if cams:
            cam_sets[gid] = cams

    all_cams = sorted({c for cams in cam_sets.values() for c in cams})
    cam_index = {c: i for i, c in enumerate(all_cams)}
    n = len(all_cams)
    mat = np.zeros((n, n), dtype=int)

    # overlap matrix: how many global_ids appear in both cams i and j
    for cams in cam_sets.values():
        cams_list = list(cams)
        for c1, c2 in combinations(cams_list, 2):
            i = cam_index[c1]
            j = cam_index[c2]
            mat[i, j] += 1
            mat[j, i] += 1
        # also count self (appear in exactly this cam)
        for c in cams_list:
            i = cam_index[c]
            mat[i, i] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, origin="lower")
    plt.colorbar(label="# Global IDs shared")
    plt.xticks(range(n), all_cams, rotation=45, ha="right")
    plt.yticks(range(n), all_cams)
    plt.title("Camera–camera overlap via global IDs")
    out_path = out_dir / "globalid_camera_overlap_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path}")

    # Histogram: how many cameras per global ID
    cams_per_gid = [len(cams) for cams in cam_sets.values()]
    plt.figure()
    plt.hist(cams_per_gid, bins=np.arange(1, max(cams_per_gid) + 2) - 0.5, rwidth=0.8)
    plt.xlabel("# cameras per global ID")
    plt.ylabel("Count of global IDs")
    plt.title("Distribution of cross-camera coverage per global ID")
    out_path = out_dir / "globalid_camera_count_hist.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot unsupervised metrics for a single DaRA AP pipeline configuration."
    )
    parser.add_argument(
        "--ap-root",
        type=str,
        required=True,
        help="AP root directory (where per_video.csv, per_track.csv live).",
    )
    parser.add_argument(
        "--config-label",
        type=str,
        default="PipelineConfig",
        help="Label to use in plot titles/legends (e.g. 'YOLOv8_AP_StrongSORT_OSNet').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Subdirectory under ap-root to write plots.",
    )
    parser.add_argument(
        "--time-bin",
        type=int,
        default=50,
        help="Number of frames per time bin for time-series plots.",
    )
    parser.add_argument(
        "--max-bones-per-cam",
        type=int,
        default=10,
        help="Max bones to plot per camera for limb length time-series.",
    )
    parser.add_argument(
        "--global-file",
        type=str,
        default="merged_global_tracks_union.jsonl",
        help="Global tracks JSONL filename under ap-root for global ID visualisation.",
    )
    parser.add_argument(
        "--track-len-clip-percentile",
        type=float,
        default=95.0,
        help=(
            "Percentile (0–100] to clip track_len for the zoomed histogram. "
            "Use 100 for no clipping; default is 95."
        ),
    )
    parser.add_argument(
        "--track-len-log-y",
        action="store_true",
        help="If set, use log scale on the y-axis for track length histograms.",
    )
    args = parser.parse_args()

    root = Path(args.ap_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(root)
    per_video = tables["per_video"]
    per_track = tables["per_track"]
    limb_df = tables["limb"]
    jitter_df = tables["jitter"]
    per_frame = tables["per_frame"]

    # Global summary
    plot_detector_coverage_vs_persistence(per_video, args.config_label, out_dir)
    plot_limb_length_cv_per_video(per_video, args.config_label, out_dir)
    plot_jitter_boxplot_by_cam(per_video, args.config_label, out_dir)
    plot_track_length_hist(
        per_track,
        args.config_label,
        out_dir,
        clip_percentile=args.track_len_clip_percentile,
        log_y=args.track_len_log_y,
    )

    # Per-camera time series
    plot_detector_tracking_timeseries_per_camera(
        per_frame, args.config_label, out_dir, time_bin=args.time_bin
    )
    plot_jitter_timeseries_per_camera(
        jitter_df, args.config_label, out_dir, time_bin=args.time_bin
    )
    plot_limb_length_timeseries_per_camera(
        limb_df, args.config_label, out_dir,
        time_bin=args.time_bin,
        max_bones_per_cam=args.max_bones_per_cam,
    )

    # Global ID visualisation
    global_path = root / args.global_file
    plot_globalid_camera_overlap(global_path, out_dir)


if __name__ == "__main__":
    main()
