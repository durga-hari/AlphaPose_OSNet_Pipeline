# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from AlphaPose_OSNet_Pipeline.Pipeline.alphapose_wrapper import AlphaPoseRunner
from AlphaPose_OSNet_Pipeline.Pipeline.osnet_wrapper import OSNetExtractor
from AlphaPose_OSNet_Pipeline.Pipeline.yolov8_wrapper import YOLOv8Detector
from AlphaPose_OSNet_Pipeline.Pipeline.strongsort_tracker import StrongSORTTracker

from AlphaPose_OSNet_Pipeline.utils.io_utils import (
    ensure_dir,
    video_writer,
    jsonl_writer,
    safe_stem,
)

# centralized visualizer
from AlphaPose_OSNet_Pipeline.utils.visualizer import (
    draw_bbox_and_id,
    draw_skeletons,
)

from datetime import datetime
import logging

log = logging.getLogger(__name__)


@dataclass
class VizCfg:
    # Visual style
    center: Tuple[int, int, int] = (0, 255, 255)   # yellow center dot
    kpt_thresh: float = 0.0                        # permissive while verifying
    dataset: str = "coco"                          # COCO-17 links
    # Geometry / scaling
    was_cropped: bool = False                      # per-person crops
    pose_input_wh: Optional[Tuple[int, int]] = None  # (W,H) — your logs showed H≈256


class InferenceRunner:
    def __init__(
        self,
        alphapose: AlphaPoseRunner,
        osnet: Optional[OSNetExtractor] = None,
        tracker: Optional[StrongSORTTracker] = None,
        detector: str = "alphapose",  # "alphapose" | "yolov8"
        y8: Optional[YOLOv8Detector] = None,
        draw: bool = True,
        save_embeds: bool = False,
        log_every: int = 50,
        viz_cfg: Optional[VizCfg] = None,
    ):
        self.ap = alphapose
        self.osn = osnet
        self.trk = tracker
        self.detector = (detector or "alphapose").lower().strip()
        self.y8 = y8
        self.draw = bool(draw)
        self.save_embeds = bool(save_embeds)
        self.log_every = int(log_every)
        self._log_counter = 0
        self.viz = viz_cfg or VizCfg()

    # ------------------------------- public API -------------------------------

    def run_on_video(
        self,
        video_path: Path,
        out_dir: Path,
        cam_id: str,
        out_video_path: Optional[Path] = None,
        out_jsonl_path: Optional[Path] = None,
        max_frames: Optional[int] = None,
        save_video: bool = True,
        save_json: bool = True,
    ) -> None:
        video_path = Path(video_path)
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error("Failed to open video: %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Outputs
        if out_video_path is None:
            out_video_path = out_dir / f"{safe_stem(video_path)}_annot.mp4"
        if out_jsonl_path is None:
            out_jsonl_path = out_dir / f"{safe_stem(video_path)}_poses.jsonl"

        vw = None
        if save_video:
            vw = video_writer(out_video_path, fps=fps, size_hw=(W, H))
            if not vw or not vw.isOpened():
                log.error("Could not initialize VideoWriter at %s", out_video_path)
                vw = None

        jsonl = None
        if save_json:
            jsonl = jsonl_writer(out_jsonl_path)

        log.info("----- RUN | cam_id=%s | video=%s | pose_in=%s (cropped=%s) -----",
                 cam_id, str(video_path), str(self.viz.pose_input_wh), self.viz.was_cropped)

        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                result = self.process_frame(frame_bgr)
                if result is None:
                    poses, embeds, boxes = [], None, np.empty((0, 4), dtype=np.float32)
                else:
                    poses, embeds, boxes = result

                if self.draw:
                    annotated = self._draw_with_visualizer(frame_bgr.copy(), poses, boxes)
                else:
                    annotated = frame_bgr

                if jsonl is not None:
                    self._write_jsonl_record(jsonl, frame_idx, cam_id, poses, embeds)

                if vw is not None:
                    vw.write(annotated)

                self._log_counter += 1
                if self._log_counter % self.log_every == 0:
                    log.info("Processed %d frames", frame_idx)

        finally:
            if jsonl is not None:
                jsonl.close()
            if vw is not None:
                vw.release()
            cap.release()
            log.info("Finished: %s | frames=%d", str(video_path), frame_idx)

    # ------------------------------- internals --------------------------------

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Returns:
          poses: list[dict] with id, bbox, keypoints, score
          embeds: np.ndarray | None
          boxes_xyxy: np.ndarray shape [N,4] float32
        """
        # 1) DETECTION
        boxes_xyxy = None
        if self.detector == "yolov8":
            if self.y8 is None:
                log.error("YOLOv8 detector selected but not provided.")
                return None
            boxes_xyxy = self.y8.detect(frame_bgr)  # may be np.ndarray or list
        else:
            if hasattr(self.ap, "detect"):
                boxes_xyxy = self.ap.detect(frame_bgr)
            else:
                return None

        boxes_xyxy = self._to_np_boxes(boxes_xyxy)
        if boxes_xyxy.shape[0] == 0:
            return None  # no detections this frame

        # 2) POSE
        kpts, pose_scores = self.ap.pose_on_boxes(frame_bgr, boxes_xyxy)
        # --- Pose sanity: per-joint uniqueness & conf ---
        if kpts:
            arr = np.asarray(kpts[0], dtype=np.float32)
            ux = np.unique(np.round(arr[:,0],3)).size
            uy = np.unique(np.round(arr[:,1],3)).size
            mc = float(arr[:,2].mean()) if arr.shape[1] >= 3 else 1.0
            logging.getLogger(__name__).warning("POSE UNIQUENESS K=%d uniq_x=%d uniq_y=%d mean_conf=%.3f",
                                                arr.shape[0], ux, uy, mc)


        # try:
        #     arr = np.asarray(kpts[0], dtype=np.float32)
        #     log.warning("POSE SHAPE=%s  min=%.2f  max=%.2f  mean_conf=%.3f",
        #                 arr.shape, float(np.nanmin(arr[..., :2])),
        #                 float(np.nanmax(arr[..., :2])),
        #                 float(arr[..., 2].mean() if arr.shape[-1] >= 3 else 1.0))
        # except Exception as e:
        #     log.warning("POSE DEBUG failed: %s", e)

        # 3) REID (optional) + TRACK (optional)
        embeds = None
        if self.osn is not None:
            if hasattr(self.osn, "embed"):
                embeds = self.osn.embed(frame_bgr, boxes_xyxy)      # [N, D]
            elif hasattr(self.osn, "extract"):
                embeds = self.osn.extract(frame_bgr, boxes_xyxy)    # [N, D]
            else:
                log.warning("OSNetExtractor has neither .embed nor .extract; skipping REID.")

        track_ids = None
        if self.trk is not None:
            track_ids = self.trk.update(boxes_xyxy, embeds)

        # 4) Build pose dicts
        poses = []
        N = boxes_xyxy.shape[0]
        for i in range(N):
            pid = int(track_ids[i]) if track_ids is not None else int(i + 1)
            poses.append({
                "id": pid,
                "bbox": [float(x) for x in boxes_xyxy[i].tolist()],
                "keypoints": kpts[i].tolist(),  # [[x,y,c], ...] in CROP coordinates
                "score": float(pose_scores[i]),
            })

        return poses, embeds, boxes_xyxy

    def _to_np_boxes(self, boxes) -> np.ndarray:
        """
        Normalize various box formats into a (N, 4) float32 np.ndarray [x1,y1,x2,y2].
        """
        if boxes is None:
            return np.empty((0, 4), dtype=np.float32)

        try:
            import torch  # noqa
            _torch_available = True
        except Exception:
            _torch_available = False

        out = []

        if isinstance(boxes, np.ndarray):
            arr = boxes
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 4 and arr.size >= 4:
                flat = arr.reshape(-1)
                k = (flat.size // 4) * 4
                flat = flat[:k]
                arr = flat.reshape(-1, 4)
            if arr.shape[1] >= 4:
                return arr[:, :4].astype(np.float32, copy=False)
            return np.empty((0, 4), dtype=np.float32)

        try:
            iterable = list(boxes)
        except TypeError:
            return np.empty((0, 4), dtype=np.float32)

        for b in iterable:
            v = None

            if _torch_available and 'torch' in str(type(b)):
                try:
                    v = b.detach().cpu().numpy()
                except Exception:
                    try:
                        v = b.cpu().numpy()
                    except Exception:
                        v = None

            if v is None and isinstance(b, (list, tuple, np.ndarray)):
                v = np.asarray(b)

            if v is None and isinstance(b, dict):
                for key in ('xyxy', 'bbox', 'box'):
                    if key in b:
                        v = b[key]
                        break
                if v is not None:
                    v = np.asarray(v)

            if v is None and hasattr(b, 'xyxy'):
                try:
                    v = np.asarray(getattr(b, 'xyxy'))
                except Exception:
                    v = None

            if v is None:
                continue

            v = np.asarray(v)
            if v.ndim == 0:
                continue
            if v.ndim == 1:
                if v.size >= 4:
                    out.append(v[:4])
            else:
                if v.shape[-1] >= 4:
                    if v.ndim > 1:
                        v = v.reshape(-1, v.shape[-1])
                    out.append(v[0, :4])

        if not out:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    # ---------------------- visualization via visualizer.py --------------------

    def _draw_with_visualizer(self, img, poses, boxes_xyxy):
        if boxes_xyxy is None:
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)

        H, W = img.shape[:2]
        total_drawn = 0

        for i, p in enumerate(poses):
            box = boxes_xyxy[i].tolist()
            pid = p.get("id", i + 1)
            score = p.get("score", None)

            # draw bbox + id
            draw_bbox_and_id(img, box, track_id=pid, score=score)

            # center dot (optional visual sanity)
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) * 0.5); cy = int((y1 + y2) * 0.5)
            cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

            # keypoints are already in IMAGE SPACE from alphapose_wrapper
            arr = np.asarray(p.get("keypoints", []), dtype=np.float32)

            # if joints are (K,2), add confidence=1.0 (optional)
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float32)], axis=1)

            diag = draw_skeletons(
                img,
                arr,
                frame_wh=None,
                input_wh=None,
                bbox_xyxy=None,
                dataset="coco",
                kpt_thresh=0.0,          # increase later (e.g., 0.1–0.2)
                pose_score=score,
                pose_score_thresh=0.0,
                was_cropped=False,
            )
            total_drawn += diag.get("n_drawn", 0)

        if total_drawn == 0 and len(poses) > 0:
            cv2.putText(img, "POSE DIAG: 0 drawn (check SPPE decode / conf)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        return img


    # --------------------------------- JSONL ----------------------------------

    def _write_jsonl_record(self, jsonl, frame_idx: int, cam_id: str, poses: List[dict], embeds):
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "frame": frame_idx,
            "cam_id": cam_id,
            "poses": poses,
        }
        if self.save_embeds and embeds is not None:
            try:
                import torch
                if isinstance(embeds, torch.Tensor):
                    embeds = embeds.detach().cpu().numpy()
            except ImportError:
                pass

            if isinstance(embeds, list):
                embeds = [e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e) for e in embeds]
                try:
                    embeds = np.vstack(embeds)
                except Exception:
                    embeds = np.array(embeds, dtype=np.float32)

            if not isinstance(embeds, np.ndarray):
                embeds = np.asarray(embeds, dtype=np.float32)

            rec["embeds"] = embeds.tolist()
        jsonl.write(json.dumps(rec) + "\n")
