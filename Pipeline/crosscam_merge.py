#!/usr/bin/env python3
"""
Cross-Camera Global Track Merger v2
-----------------------------------
Merges per-camera RTMO pose-tracking outputs into one
unified JSONL file with globally consistent person IDs.

Requires:
  - crosscam_map.json (from crosscam_stitch_v2.py)
  - results.jsonl files per camera
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse

def load_crosscam_map(map_path):
    """Load global mapping dict of camera→localID→globalID."""
    with open(map_path, "r") as f:
        return json.load(f)

def merge_tracks(root, crosscam_map, out_path):
    """Merge all per-camera results.jsonl into unified global results."""
    out = []
    root = Path(root)
    jsonl_files = sorted(root.glob("*/results.jsonl"))
    if not jsonl_files:
        print(f"[ERROR] No results.jsonl found under {root}")
        return

    for fpath in jsonl_files:
        cam_name = fpath.parent.name
        print(f"[INFO] Processing camera: {cam_name}")
        local_to_global = crosscam_map.get(cam_name, {})

        with open(fpath, "r") as f:
            for line in tqdm(f, desc=f"{cam_name}"):
                item = json.loads(line)
                local_id = str(item.get("id"))
                global_id = local_to_global.get(local_id, None)
                if global_id is None:
                    continue  # skip unlinked IDs

                merged_item = {
                    "global_id": global_id,
                    "camera": cam_name,
                    "frame": item.get("frame"),
                    "timestamp": item.get("ts", None),
                    "bbox": item.get("bbox", None),
                    "keypoints": item.get("keypoints", None),
                    "embedding": item.get("embedding", None) or item.get("emb", None),
                    "score": item.get("score", None)
                }
                out.append(merged_item)

    print(f"[INFO] Writing merged file → {out_path}")
    with open(out_path, "w") as fw:
        for rec in out:
            fw.write(json.dumps(rec) + "\n")
    print(f"[DONE] Merged {len(out)} records into unified global timeline.")


def main(args):
    root = Path(args.root)
    map_path = root / "crosscam_map.json"
    out_path = root / "merged_global_tracks.jsonl"

    if not map_path.exists():
        print(f"[ERROR] crosscam_map.json not found at {map_path}")
        return

    crosscam_map = load_crosscam_map(map_path)
    merge_tracks(root, crosscam_map, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-camera RTMO outputs into global track file.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing camera subfolders with results.jsonl and crosscam_map.json")
    args = parser.parse_args()
    main(args)
