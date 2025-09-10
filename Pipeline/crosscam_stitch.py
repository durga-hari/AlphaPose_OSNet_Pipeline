#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
crosscam_stitch.py

Reads per-camera JSONL pose outputs (that include "emb": vector when available),
computes an average embedding per (cam_id, track_id), then merges tracks across
cameras by cosine similarity and time overlap.

Usage:
  python -m AlphaPose_OSNet_Pipeline.Pipeline.crosscam_stitch \
    --root /home/athena/DaRA_Thesis/Output_AP_OSNET \
    --sim-thr 0.55 --time-win 900 \
    --out /home/athena/DaRA_Thesis/Output_AP_OSNET/crosscam_map.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

def _cos(a,b): 
    if a is None or b is None: return -1.0
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    na=np.linalg.norm(a)+1e-12; nb=np.linalg.norm(b)+1e-12
    return float(np.dot(a/na, b/nb))

def _read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_track_embeds(cam_dir: Path) -> Dict[int, dict]:
    """
    Returns: track_id -> {"emb": mean_vec or None, "t0": min_frame, "t1": max_frame}
    """
    # expect exactly one *_poses.jsonl in cam_dir
    jls=list(cam_dir.glob("*_poses.jsonl"))
    if not jls:
        return {}
    jl=jls[0]
    tr: Dict[int, dict]={}
    for rec in _read_jsonl(jl):
        t=rec.get("frame", None)
        for p in rec.get("poses", []):
            pid=int(p.get("id", -1))
            emb=p.get("emb", None)
            if pid<0: continue
            if pid not in tr: tr[pid]={"sum":None,"n":0,"t0":t,"t1":t}
            if emb is not None:
                v=np.asarray(emb,dtype=np.float32).reshape(-1)
                if tr[pid]["sum"] is None: tr[pid]["sum"]=v.copy()
                else: tr[pid]["sum"]+=v
                tr[pid]["n"]+=1
            tr[pid]["t0"]=min(tr[pid]["t0"], t); tr[pid]["t1"]=max(tr[pid]["t1"], t)
    # finalize mean
    out={}
    for pid, d in tr.items():
        if d["sum"] is None or d["n"]==0: m=None
        else: m=d["sum"]/float(d["n"])
        out[pid]={"emb": (m.tolist() if m is not None else None), "t0": d["t0"], "t1": d["t1"]}
    return out

def stitch(root: Path, sim_thr: float, time_win: int) -> Dict[str, Dict[int,int]]:
    """
    Returns mapping: cam_id -> { local_id -> global_id }
    """
    cams=[d for d in root.iterdir() if d.is_dir()]
    # Per-cam track summaries
    per_cam={}
    for c in cams:
        per_cam[c.name]=build_track_embeds(c)

    # Assign global IDs by greedy cosine match across cams
    next_gid=1
    global_map: Dict[str, Dict[int,int]]={c:{} for c in per_cam.keys()}
    gid_emb: Dict[int, np.ndarray]={}
    gid_time: Dict[int, Tuple[int,int]]={}
    # Flatten all tracks
    all_items=[]
    for cam, tracks in per_cam.items():
        for tid, info in tracks.items():
            all_items.append((cam, tid, info))
    # sort by earliest time
    all_items.sort(key=lambda x: (x[2]["t0"] if x[2]["t0"] is not None else 0))

    for cam, tid, info in all_items:
        emb=info["emb"]; t0,t1=info["t0"], info["t1"]
        if emb is None:
            # no embedding -> new global id
            gid=next_gid; next_gid+=1
            global_map[cam][tid]=gid
            gid_emb[gid]=None
            gid_time[gid]=(t0,t1)
            continue

        # find best existing gid by cosine + temporal overlap
        best_gid=None; best=-1.0
        for gid, gemb in gid_emb.items():
            if gemb is None: continue
            # time gate: allow within time_win seconds (assuming FPS ~ 25 -> convert if you want)
            gt0, gt1 = gid_time[gid]
            if (t0 is not None and gt1 is not None and t0 - gt1 > time_win) or \
               (gt0 is not None and t1 is not None and gt0 - t1 > time_win):
                continue
            s=_cos(emb, gemb)
            if s>best: best=s; best_gid=gid
        if best_gid is not None and best >= sim_thr:
            global_map[cam][tid]=best_gid
            # update running mean emb and time span
            gemb=gid_emb[best_gid]
            m = (np.asarray(gemb)+np.asarray(emb))/2.0 if gemb is not None else np.asarray(emb)
            gid_emb[best_gid]=m
            gt0,gt1=gid_time[best_gid]
            gid_time[best_gid]=(min(gt0,t0), max(gt1,t1))
        else:
            gid=next_gid; next_gid+=1
            global_map[cam][tid]=gid
            gid_emb[gid]=np.asarray(emb) if emb is not None else None
            gid_time[gid]=(t0,t1)

    return global_map

def main():
    ap=argparse.ArgumentParser(description="Cross-camera ID stitcher (embedding-based)")
    ap.add_argument("--root", required=True, help="Output root containing per-camera folders with *_poses.jsonl")
    ap.add_argument("--sim-thr", type=float, default=0.55, help="Cosine similarity threshold")
    ap.add_argument("--time-win", type=int, default=900, help="Temporal window for matching (frames or seconds; tune for your data)")
    ap.add_argument("--out", required=True, help="Where to write the mapping JSON")
    args=ap.parse_args()

    root=Path(args.root)
    mapping=stitch(root, sim_thr=args.sim_thr, time_win=args.time_win)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote cross-camera mapping to: {args.out}")

if __name__ == "__main__":
    main()
