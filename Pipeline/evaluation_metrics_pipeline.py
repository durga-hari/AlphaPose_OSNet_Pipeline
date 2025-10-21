#!/usr/bin/env python3
"""
Evaluate AlphaPose-OSNet pipeline performance *without ground truth*.

Finds JSONL results such as:
    /home/arun_remote/DaRA_Thesis/Output_AP_OSNET/AC10/AC10_poses.jsonl

Computes:
 - Pose stability & bone-length consistency
 - Re-ID embedding coherence
 - Track lifetime & fragmentation
 - FPS and composite efficiency indices

Outputs:
    /home/arun_remote/DaRA_Thesis/AlphaPose_OSNet_Pipeline/reports/metrics_unsupervised/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ---------- PATH SETTINGS ----------
RESULTS_ROOT = Path("/home/arun_remote/DaRA_Thesis/Output_AP_OSNET")
REPORT_DIR = Path("/home/arun_remote/DaRA_Thesis/Output_AP_OSNET/reports/metrics_unsupervised")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- BODY CONNECTIVITY ----------
SKELETON_PAIRS = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

# ---------- HELPERS ----------
def load_jsonl(path: Path):
    """Read JSONL file line-by-line."""
    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
    return data

def compute_pose_stability(track_frames):
    coords = np.array(track_frames)
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=2)
    return float(np.exp(-np.nanmean(diffs)))

def compute_bone_consistency(track_frames):
    coords = np.array(track_frames)
    bone_lengths = []
    for a, b in SKELETON_PAIRS:
        if a < coords.shape[1] and b < coords.shape[1]:
            dist = np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1)
            bone_lengths.append(np.nanstd(dist))
    return float(np.exp(-np.nanmean(bone_lengths)))

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_embedding_consistency(embs):
    if len(embs) < 2:
        return np.nan
    sims = [cosine_similarity(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    return float(np.nanmean(sims))

def compute_track_metrics(tracks):
    lifetimes = [len(frames) for frames in tracks.values()]
    frag = np.sum([len(frames) == 1 for frames in tracks.values()]) / max(len(tracks), 1)
    return np.mean(lifetimes), frag

# ---------- CORE EVALUATION ----------
def evaluate_file(jsonl_path: Path):
    data = load_jsonl(jsonl_path)
    if not data:
        print(f"[WARN] Empty or invalid {jsonl_path}")
        return None

    tracks, frame_times = {}, []
    for item in data:
        tid = item.get("idx", -1)
        kps = np.array(item.get("keypoints", []), dtype=float).reshape(-1, 3)[:, :2]
        emb = np.array(item.get("feature", []), dtype=float)
        ts = item.get("timestamp")
        if ts:
            frame_times.append(ts)
        tracks.setdefault(tid, {"frames": [], "embs": []})
        tracks[tid]["frames"].append(kps)
        if emb.size > 0:
            tracks[tid]["embs"].append(emb)

    pose_stab, bone_cons, emb_cons = [], [], []
    for tid, vals in tqdm(tracks.items(), desc=f"[INFO] Evaluating {jsonl_path.parent.name}"):
        if len(vals["frames"]) < 2:
            continue
        pose_stab.append(compute_pose_stability(vals["frames"]))
        bone_cons.append(compute_bone_consistency(vals["frames"]))
        if vals["embs"]:
            emb_cons.append(compute_embedding_consistency(vals["embs"]))

    tl, frag = compute_track_metrics(tracks)
    fps = np.nan
    if frame_times:
        frame_times = np.sort(np.array(frame_times))
        d = np.diff(frame_times)
        if np.nanmean(d) > 0:
            fps = 1.0 / np.nanmean(d)

    metrics = {
        "video_name": jsonl_path.stem.replace("_poses", ""),
        "pose_stability": np.nanmean(pose_stab),
        "bone_consistency": np.nanmean(bone_cons),
        "embedding_consistency": np.nanmean(emb_cons),
        "avg_track_lifetime": tl,
        "frag_ratio": frag,
        "avg_fps": fps,
        "num_tracks": len(tracks),
        "total_frames": len(data),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics["pose_stability_index"] = np.nanmean(
        [metrics["pose_stability"], metrics["bone_consistency"]]
    )
    metrics["reid_coherence_score"] = metrics["embedding_consistency"]
    metrics["tracking_reliability_index"] = (1 - metrics["frag_ratio"]) * (
        metrics["avg_track_lifetime"] / (metrics["avg_track_lifetime"] + 10)
    )
    metrics["pipeline_efficiency_score"] = (metrics["avg_fps"] / 30.0) * (
        1 - metrics["frag_ratio"]
    )
    return metrics

# ---------- RUNNER ----------
def main():
    print(f"[INFO] Recursively scanning under {RESULTS_ROOT} ...")
    jsonl_files = sorted(RESULTS_ROOT.rglob("*_poses.jsonl"))
    print(f"[DEBUG] Found {len(jsonl_files)} files:")
    for f in jsonl_files:
        print("   ", f)

    if not jsonl_files:
        print(f"[ERROR] No *_poses.jsonl files found under {RESULTS_ROOT}")
        return

    all_metrics = []
    for path in jsonl_files:
        m = evaluate_file(path)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        print("[WARN] No metrics computed.")
        return

    df = pd.DataFrame(all_metrics)
    csv_path = REPORT_DIR / "unsupervised_metrics_summary.csv"
    md_path = REPORT_DIR / "unsupervised_metrics_summary.md"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"\n Metrics saved:\n  {csv_path}\n  {md_path}\n")
    print(df.round(3))


if __name__ == "__main__":
    main()
