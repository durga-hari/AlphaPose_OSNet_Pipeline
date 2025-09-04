# ================= pipeline/tracker.py =================
import numpy as np
from collections import defaultdict

class MultiCameraPersonTracker:
    def __init__(self, config):
        self.config = config
        self.next_global_id = 0
        self.tracks = defaultdict(dict)  # {camera_id: {track_id: {bbox, feature}}}
        self.global_track_map = {}  # {local_track_id: global_id}

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _associate_global_id(self, camera_id, local_id, feature):
        best_sim = 0.0
        best_gid = None
        for cid, cam_tracks in self.tracks.items():
            for tid, data in cam_tracks.items():
                if f"{cid}_{tid}" == f"{camera_id}_{local_id}":
                    continue
                sim = self._cosine_similarity(data['feature'], feature)
                if sim > 0.7 and sim > best_sim:
                    best_sim = sim
                    best_gid = self.global_track_map.get(f"{cid}_{tid}")
        if best_gid is None:
            best_gid = self.next_global_id
            self.next_global_id += 1
        self.global_track_map[f"{camera_id}_{local_id}"] = best_gid
        return best_gid

    def update(self, camera_id, bboxes, poses, features):
        results = {}
        for i, bbox in enumerate(bboxes):
            local_id = f"{camera_id}_{i}"
            self.tracks[camera_id][local_id] = {
                'bbox': bbox,
                'pose': poses[i],
                'feature': features[i]
            }
            global_id = self._associate_global_id(camera_id, i, features[i])
            results[global_id] = self.tracks[camera_id][local_id]
        return results