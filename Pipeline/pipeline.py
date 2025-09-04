
from __future__ import annotations
import os, json, csv, time, signal
import cv2
import numpy as np
import torch

EXIT_REQUESTED = {"flag": False}
def _handle_sigint(signum, frame):
    EXIT_REQUESTED["flag"] = True
signal.signal(signal.SIGINT, _handle_sigint)

# pipeline/pipeline.py
from .alphapose_wrapper import load_alphapose
from .osnet_wrapper import load_osnet
from .tracker import MultiCameraPersonTracker
from .inference_runner import run_inference
from ..utils.visualizer import draw_bbox_and_id, draw_skeleton

# ...
# Output writer should use Config.output_dir, which is now: /home/athena/DaRA_Thesis/Output_AP_OSNET


# ---------- helpers ----------
def _to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _safe_crop(frame, bb_xyxy):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bb_xyxy[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2].copy()
    return crop, [0.0, 0.0, float(x2 - x1), float(y2 - y1)]

def _normalize_pose_output(p):
    if p is None:
        return None
    if isinstance(p, dict):
        for k in ("keypoints", "kpts", "pts", "preds"):
            if k in p and p[k] is not None:
                return p[k]
        if "result" in p and isinstance(p["result"], (list, tuple)) and p["result"]:
            r0 = p["result"][0]
            if isinstance(r0, dict) and "keypoints" in r0:
                return r0["keypoints"]
    if hasattr(p, "keypoints"):
        return getattr(p, "keypoints")
    if isinstance(p, (list, tuple, np.ndarray)):
        return p
    return None

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
# -----------------------------

class MultiCamOutputWriter:
    def __init__(self, out_root: str, session_tag: str):
        self.root = out_root
        self.dir_videos = os.path.join(out_root, "videos")
        self.dir_tracks = os.path.join(out_root, "tracks")
        self.dir_poses = os.path.join(out_root, "poses")
        for d in [out_root, self.dir_videos, self.dir_tracks, self.dir_poses]:
            os.makedirs(d, exist_ok=True)
        self.track_csv, self.track_fh, self.pose_fh = {}, {}, {}
        self.manifest = {"session_tag": session_tag, "created_utc": int(time.time()),
                         "cameras": {}, "files": {}}
        self.global_appear = {}

    def register_camera(self, cam_id: str, fps: float, width: int, height: int, annotated_path: str):
        csv_path = os.path.join(self.dir_tracks, f"{cam_id}_tracks.csv")
        fh = open(csv_path, "w", newline="", encoding="utf-8")
        wr = csv.writer(fh)
        wr.writerow(["frame", "timestamp_ms", "cam_id", "gid", "x1", "y1", "x2", "y2", "det_conf"])
        self.track_csv[cam_id], self.track_fh[cam_id] = wr, fh

        pose_path = os.path.join(self.dir_poses, f"{cam_id}_poses.jsonl")
        self.pose_fh[cam_id] = open(pose_path, "w", encoding="utf-8")

        self.manifest["cameras"][cam_id] = {"fps": fps, "width": width, "height": height}
        self.manifest["files"][cam_id] = {
            "video_annotated": annotated_path,
            "tracks_csv": csv_path,
            "poses_jsonl": pose_path
        }

    def write_tracks(self, cam_id: str, frame_idx: int, ts_ms: int, tracks: dict):
        wr = self.track_csv[cam_id]
        for gid, t in tracks.items():
            x1, y1, x2, y2, conf = t["bbox"]
            wr.writerow([frame_idx, ts_ms, cam_id, gid, int(x1), int(y1), int(x2), int(y2), float(conf)])
            self._update_appearance(gid, cam_id, frame_idx)

    def write_poses(self, cam_id: str, frame_idx: int, poses_by_gid: dict, write_empty=True):
        fh = self.pose_fh[cam_id]
        if not poses_by_gid:
            if write_empty:
                fh.write(json.dumps({"frame": frame_idx, "cam_id": cam_id, "poses": []}) + "\n")
            return
        for gid, kpts in poses_by_gid.items():
            fh.write(json.dumps({"frame": frame_idx, "cam_id": cam_id, "gid": int(gid), "keypoints": kpts}) + "\n")

    def _update_appearance(self, gid: int, cam_id: str, frame_idx: int):
        g = self.global_appear.setdefault(int(gid), {})
        lst = g.setdefault(cam_id, [])
        if not lst or lst[-1]["end"] < frame_idx - 1:
            lst.append({"start": frame_idx, "end": frame_idx})
        else:
            lst[-1]["end"] = frame_idx

    def finalize(self):
        for fh in self.track_fh.values():
            fh.close()
        for fh in self.pose_fh.values():
            fh.close()
        with open(os.path.join(self.root, "global_tracks.json"), "w", encoding="utf-8") as f:
            json.dump(self.global_appear, f, indent=2)
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def flush(self, cam_id: str | None = None):
        if cam_id is not None:
            if cam_id in self.track_fh: self.track_fh[cam_id].flush()
            if cam_id in self.pose_fh:
                self.pose_fh[cam_id].flush()
                try: os.fsync(self.pose_fh[cam_id].fileno())
                except Exception: pass
            return
        for fh in self.track_fh.values(): fh.flush()
        for fh in self.pose_fh.values():
            fh.flush()
            try: os.fsync(fh.fileno())
            except Exception: pass

    def checkpoint_poses(self, cam_id: str, max_lines: int = 200000):
        src = os.path.join(self.dir_poses, f"{cam_id}_poses.jsonl")
        dst_tmp = os.path.join(self.dir_poses, f"{cam_id}_poses.json.tmp")
        dst     = os.path.join(self.dir_poses, f"{cam_id}_poses.json")
        arr = []
        try:
            with open(src, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue
                    try: arr.append(json.loads(line))
                    except Exception: continue
                    if i >= max_lines: break
            with open(dst_tmp, "w", encoding="utf-8") as g:
                json.dump(arr, g, ensure_ascii=False)
            os.replace(dst_tmp, dst)
        except FileNotFoundError:
            return
        except Exception:
            return


class DaRAPipeline:
    def __init__(self, config, logger, session_tag: str = "run"):
        self.cfg = config
        self.logger = logger
        self.session_tag = session_tag
        self.detector, self.pose_model, self.run_pose = load_alphapose(config)
        self.reid_model, self.reid_tf = load_osnet(config)
        self.tracker = MultiCameraPersonTracker(config)
        self.out = MultiCamOutputWriter(config.output_dir, session_tag)

        # Loosen detector thresholds if supported
        for attr, val in [
            ("conf_thres", float(getattr(self.cfg, "det_conf_threshold", 0.25))),
            ("confidence", float(getattr(self.cfg, "det_conf_threshold", 0.25))),
            ("score_thresh", float(getattr(self.cfg, "det_conf_threshold", 0.25))),
            ("nms_thres", float(getattr(self.cfg, "det_nms_threshold", 0.45))),
        ]:
            if hasattr(self.detector, attr):
                try:
                    setattr(self.detector, attr, val)
                    self.logger.info(f"[DET] set {attr}={val}")
                except Exception:
                    pass

        # Pass class names if wrapper supports it
        if hasattr(self.cfg, "yolo_names") and os.path.isfile(self.cfg.yolo_names):
            try:
                if hasattr(self.detector, "class_names"):
                    with open(self.cfg.yolo_names, "r", encoding="utf-8") as f:
                        self.detector.class_names = [ln.strip() for ln in f if ln.strip()]
                        self.logger.info(f"[DET] loaded class names ({len(self.detector.class_names)})")
            except Exception:
                pass

    def _detect(self, proc):
        # Normalize input to BGR numpy frame
        if isinstance(proc, np.ndarray):
            frame_bgr = proc
        elif hasattr(proc, "frame"):
            frame_bgr = proc.frame
        elif isinstance(proc, dict) and "frame" in proc:
            frame_bgr = proc["frame"]
        else:
            raise TypeError("Unsupported 'proc' type")
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Preprocess
        try:
            prepped = self.detector.image_preprocess(frame_bgr)
        except Exception:
            prepped = self.detector.image_preprocess([frame_bgr])

        # Normalize preprocess outputs
        im_batches = None; im_dim_list = None; orig_dim_list = None
        if isinstance(prepped, (torch.Tensor, np.ndarray)):
            im_batches = prepped
        elif isinstance(prepped, (list, tuple)):
            if len(prepped) == 2:
                im_batches, im_dim_list = prepped
            elif len(prepped) == 3:
                im_batches, im_dim_list, orig_dim_list = prepped
            elif len(prepped) == 4:
                im_batches, _orig_imgs, im_dim_list, orig_dim_list = prepped
            else:
                im_batches = prepped[0]
        elif isinstance(prepped, dict) or hasattr(prepped, "__dict__"):
            getv = prepped.get if isinstance(prepped, dict) else (lambda k: getattr(prepped, k, None))
            im_batches = getv("im_batches") or getv("images") or getv("batches") or prepped
            im_dim_list = getv("im_dim_list") or getv("dims") or getv("im_dims")
            orig_dim_list = getv("orig_dim_list") or getv("original_dims") or getv("orig_dims")
        else:
            im_batches = prepped

        H, W = frame_bgr.shape[:2]
        if im_dim_list is None: im_dim_list = [(W, H)]
        if orig_dim_list is None: orig_dim_list = [(W, H)]
        if not isinstance(im_dim_list, torch.Tensor):
            im_dim_list = torch.tensor(im_dim_list, dtype=torch.float32)
        if not isinstance(orig_dim_list, torch.Tensor):
            orig_dim_list = torch.tensor(orig_dim_list, dtype=torch.float32)

        # Run detection (2-arg preferred, fallback to 3-arg)
        try:
            det_out = self.detector.images_detection(im_batches, im_dim_list)
        except TypeError:
            det_out = self.detector.images_detection(im_batches, im_dim_list, orig_dim_list)

        dets = det_out[0] if isinstance(det_out, (list, tuple)) and len(det_out) > 0 else det_out
        if dets is None:
            return []

        # Normalize detections to [x1, y1, x2, y2, score] — keep ALL classes
        boxes = []
        try:
            for d in dets:
                x1, y1, x2, y2 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                conf = float(d[4]) if len(d) > 4 else 1.0
                boxes.append([x1, y1, x2, y2, conf])
        except Exception:
            boxes = []
        return boxes

    def _extract_feats(self, frame, bboxes):
        feats = []
        for bb in bboxes:
            x1, y1, x2, y2 = map(int, bb[:4])
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0 or (x2 - x1) <= 1 or (y2 - y1) <= 1:
                feats.append(np.zeros(512, dtype=np.float32))
                continue
            t = self.reid_tf(crop).unsqueeze(0).to(self.cfg.device)
            with torch.no_grad():
                f = self.reid_model(t).cpu().numpy().flatten().astype(np.float32)
            feats.append(f)
        return feats

    def run(self):
        caps, outs, fps_map, frame_idx = {}, {}, {}, {}
        for i, path in enumerate(self.cfg.input_video_paths):
            cam_id = self.cfg.camera_ids[i]
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.logger.warning(f"[WARN] Failed to open {path}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_map[cam_id] = fps
            caps[cam_id] = cap
            frame_idx[cam_id] = 0

            out_path = os.path.join(self.out.dir_videos, f"{cam_id}_annotated.mp4")
            if self.cfg.draw_outputs:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                outs[cam_id] = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            else:
                outs[cam_id] = None

            self.out.register_camera(cam_id, fps, W, H, out_path)

        if not caps:
            self.logger.error("No inputs opened.")
            return

        t0 = time.time()
        frames_done = {cam_id: 0 for cam_id in caps.keys()}
        PROGRESS_EVERY = getattr(self.cfg, "progress_every", 200)
        CHECKPOINT_EVERY = getattr(self.cfg, "checkpoint_every", 1000)
        DEBUG_DRAW_DETS = bool(getattr(self.cfg, "debug_draw_dets", False))

        try:
            while True:
                if EXIT_REQUESTED["flag"]:
                    self.logger.info("[INFO] Ctrl+C requested — stopping after this cycle...")
                    break

                frames, timestamps = {}, {}
                for cam_id, cap in caps.items():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx[cam_id])
                    ok, frame = cap.read()
                    frames[cam_id] = frame if ok else None
                    if ok:
                        ts_ms = int(1000.0 * frame_idx[cam_id] / (fps_map[cam_id] or 30.0))
                        timestamps[cam_id] = ts_ms

                if all(f is None for f in frames.values()):
                    break

                min_frame = min(frame_idx.values())
                for cid in frame_idx:
                    frame_idx[cid] = min_frame + max(1, self.cfg.frame_skip)

                for cam_id, frame in frames.items():
                    if EXIT_REQUESTED["flag"]:
                        break
                    if frame is None:
                        continue
                    ts_ms = timestamps.get(cam_id, 0)

                    # Detector (with optional downscale)
                    s = self.cfg.detector_scale if self.cfg.detector_scale and self.cfg.detector_scale > 0 else 1.0
                    proc = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    bboxes = self._detect(proc)
                    if s != 1.0:
                        for b in bboxes:
                            b[0] /= s; b[1] /= s; b[2] /= s; b[3] /= s

                    # Detector progress debug
                    if (frames_done.get(cam_id, 0) % max(1, PROGRESS_EVERY)) == 0:
                        top5 = [round(bb[4], 3) for bb in sorted(bboxes, key=lambda x: -x[4])[:5]]
                        self.logger.info(f"[DETDBG] cam={cam_id} frame={frame_idx[cam_id]} dets={len(bboxes)} top_scores={top5}")

                    # Optional raw detection overlay to isolate detector vs tracker
                    if DEBUG_DRAW_DETS and self.cfg.draw_outputs and outs[cam_id] is not None:
                        dbg = frame.copy()
                        for x1, y1, x2, y2, sc in bboxes:
                            cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                            cv2.putText(dbg, f"{sc:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        outs[cam_id].write(dbg)
                        self.out.write_tracks(cam_id, frame_idx[cam_id], ts_ms, {})
                        self.out.write_poses(cam_id, frame_idx[cam_id], {}, write_empty=True)
                        frames_done[cam_id] += 1
                        continue

                    # Pose: robust calling (RGB + crop/full, XYWH/XYXY) + normalization
                    poses = []
                    valid_pose_count = 0
                    frame_rgb = _to_rgb(frame)
                    for bb in bboxes:
                        pose_payload = None
                        try:
                            crop_bgr, bb_in_crop_xywh = _safe_crop(frame, bb)
                            if crop_bgr is not None:
                                crop_rgb = _to_rgb(crop_bgr)
                                p = self.run_pose(self.pose_model, crop_rgb, bb_in_crop_xywh)
                                pose_payload = _normalize_pose_output(p)
                            if pose_payload is None:
                                x1, y1, x2, y2 = bb[:4]
                                xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                p = self.run_pose(self.pose_model, frame_rgb, xywh)
                                pose_payload = _normalize_pose_output(p)
                            if pose_payload is None:
                                p = self.run_pose(self.pose_model, frame_rgb, [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                                pose_payload = _normalize_pose_output(p)
                        except Exception:
                            pose_payload = None

                        if pose_payload is not None:
                            valid_pose_count += 1
                        poses.append(pose_payload)

                    # >>> moved outside the loop <<<
                    if bboxes and valid_pose_count == 0:
                        self.logger.warning(
                            f"[POSEDBG] cam={cam_id} frame={frame_idx[cam_id]}: "
                            f"{len(bboxes)} boxes but all pose returns are None"
                        )
                        # TEMP: one raw-dump to see structure if all failed
                        try:
                            test_bb = bboxes[0]
                            crop_bgr, bb_in_crop_xywh = _safe_crop(frame, test_bb)
                            raw = None
                            if crop_bgr is not None:
                                raw = self.run_pose(self.pose_model, _to_rgb(crop_bgr), bb_in_crop_xywh)
                            if raw is None:
                                x1, y1, x2, y2 = test_bb[:4]
                                xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                raw = self.run_pose(self.pose_model, frame_rgb, xywh)
                            if raw is None:
                                raw = self.run_pose(self.pose_model, frame_rgb, [float(test_bb[0]), float(test_bb[1]), float(test_bb[2]), float(test_bb[3])])
                            tname = type(raw).__name__
                            keys = list(raw.keys())[:10] if isinstance(raw, dict) else "n/a"
                            self.logger.warning(f"[POSEDBG] raw return type={tname} keys={keys}")
                        except Exception as _e:
                            self.logger.warning(f"[POSEDBG] raw pose call raised: {_e}")

                    # ReID features
                    feats = self._extract_feats(frame, bboxes)

                    # Tracker
                    tracks = self.tracker.update(cam_id, bboxes, poses, feats)

                    # Fallback: attach pose by IoU if tracker ignored it
                    det_boxes = [list(map(float, bb[:4])) for bb in bboxes]
                    for gid, t in tracks.items():
                        if t.get("pose") is not None:
                            continue
                        tb = list(map(float, t["bbox"][:4]))
                        best_iou, best_idx = 0.0, -1
                        for i, db in enumerate(det_boxes):
                            iou = _iou(tb, db)
                            if iou > best_iou:
                                best_iou, best_idx = iou, i
                        if best_idx >= 0 and poses[best_idx] is not None:
                            t["pose"] = poses[best_idx]

                    # Draw/write
                    if self.cfg.draw_outputs and outs[cam_id] is not None:
                        for gid, t in tracks.items():
                            x1, y1, x2, y2, _ = t["bbox"]
                            draw_bbox_and_id(frame, [x1, y1, x2, y2], f"GID:{gid}")
                            if t.get("pose") is not None:
                                draw_skeleton(frame, t["pose"])
                        outs[cam_id].write(frame)

                    self.out.write_tracks(cam_id, frame_idx[cam_id], ts_ms, tracks)
                    poses_by_gid = {gid: t["pose"] for gid, t in tracks.items() if t.get("pose") is not None}
                    self.out.write_poses(cam_id, frame_idx[cam_id], poses_by_gid, write_empty=True)

                    # progress + checkpoints
                    frames_done[cam_id] += 1
                    if frames_done[cam_id] % PROGRESS_EVERY == 0:
                        elapsed = max(1e-6, time.time() - t0)
                        total = sum(frames_done.values())
                        fps = total / elapsed
                        self.out.flush(cam_id)
                        self.logger.info(f"[PROGRESS] cam={cam_id} frames={frames_done[cam_id]} "
                                         f"pose_ok={valid_pose_count} total={total} "
                                         f"elapsed={elapsed:.1f}s ~{fps:.2f} fps")
                    if frames_done[cam_id] % CHECKPOINT_EVERY == 0:
                        self.out.checkpoint_poses(cam_id)

                if EXIT_REQUESTED["flag"]:
                    break
        finally:
            for cap in caps.values(): cap.release()
            for w in outs.values():
                if w is not None: w.release()
            try:
                for cam_id in caps.keys():
                    self.out.checkpoint_poses(cam_id)
            except Exception:
                pass
            self.out.finalize()

            elapsed = max(1e-6, time.time() - t0)
            total = sum(frames_done.values()) if frames_done else 0
            fps = total / elapsed if elapsed > 0 else 0.0
            self.logger.info(f"[DONE] total_frames={total} elapsed={elapsed:.1f}s ~{fps:.2f} fps")
            self.logger.info(f"[DONE] outputs at: {self.out.root}")




