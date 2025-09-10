# -*- coding: utf-8 -*-

"""
strongsort_tracker.py

Lightweight StrongSORT-style tracker:
- Kalman filter (constant velocity) for motion
- IoU gating
- ReID (cosine) association with OSNet embeddings (L2-normalized)
- Track TTL to drop stale tracks

API:
  trk = StrongSORTTracker(sim_thr=0.45, iou_thr=0.3, ttl=60)
  ids = trk.update(boxes, feats)   # boxes: [[x1,y1,x2,y2], ...], feats: [np.ndarray|None, ...]
"""

from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np

def _iou(a,b) -> float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih; ua=max(0.0,(ax2-ax1)*(ay2-ay1)); ub=max(0.0,(bx2-bx1)*(by2-by1))
    union=ua+ub-inter+1e-12
    return float(inter/union)

class _KF:
    """Minimal 8D state Kalman: [cx,cy,w,h,vx,vy,vw,vh]"""
    def __init__(self, box):
        cx=(box[0]+box[2])/2.0; cy=(box[1]+box[3])/2.0; w=box[2]-box[0]; h=box[3]-box[1]
        self.x=np.array([cx,cy,w,h,0,0,0,0],dtype=np.float32)
        self.P=np.eye(8,dtype=np.float32)*10.0
        self.Q=np.eye(8,dtype=np.float32)*0.01
        self.R=np.eye(4,dtype=np.float32)*1.0
        self.H=np.zeros((4,8),dtype=np.float32); self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0
        self._F=np.eye(8,dtype=np.float32)
        for i in range(4): self._F[i,i+4]=1.0

    def predict(self):
        self.x=self._F@self.x; self.P=self._F@self.P@self._F.T+self.Q

    def update(self, box):
        z=np.array([(box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]],dtype=np.float32)
        y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S)
        self.x=self.x+K@y; self.P=(np.eye(8,dtype=np.float32)-K@self.H)@self.P

    def box(self):
        cx,cy,w,h=self.x[0],self.x[1],max(1.0,self.x[2]),max(1.0,self.x[3])
        return [float(cx-w/2.0), float(cy-h/2.0), float(cx+w/2.0), float(cy+h/2.0)]

class StrongSORTTracker:
    def __init__(self, sim_thr: float = 0.45, iou_thr: float = 0.3, ttl: int = 60):
        self.sim_thr=float(sim_thr); self.iou_thr=float(iou_thr); self.ttl=int(ttl)
        self._next_id=1
        self._tracks: Dict[int, dict]={}  # id -> {kf, feat, age}

    def _sim(self, f1, f2) -> float:
        if f1 is None or f2 is None: return -1.0
        return float(np.dot(f1, f2))  # expect L2-normalized

    def update(self, boxes: List[List[float]], feats: List[Optional[np.ndarray]]) -> List[int]:
        # predict step
        for t in self._tracks.values(): t["kf"].predict(); t["age"]+=1
        # remove stale
        for tid in [tid for tid,t in self._tracks.items() if t["age"]>self.ttl]:
            self._tracks.pop(tid,None)

        assigned=[False]*len(boxes)
        ids=[-1]*len(boxes)

        # Try to match by appearance+IoU
        for i, b in enumerate(boxes):
            best_id=None; best=-1e9
            for tid, t in self._tracks.items():
                iou=_iou(b, t["kf"].box())
                if iou < self.iou_thr: continue
                s=self._sim(feats[i] if i<len(feats) else None, t["feat"])
                if s>best: best=s; best_id=tid
            if best_id is not None and best >= self.sim_thr:
                self._tracks[best_id]["kf"].update(b)
                # keep last feat if current None
                if i<len(feats) and feats[i] is not None: self._tracks[best_id]["feat"]=feats[i]
                self._tracks[best_id]["age"]=0
                ids[i]=best_id; assigned[i]=True

        # Unmatched -> new tracks
        for i, b in enumerate(boxes):
            if assigned[i]: continue
            tid=self._next_id; self._next_id+=1
            self._tracks[tid]={"kf":_KF(b), "feat": (feats[i] if i<len(feats) else None), "age":0}
            ids[i]=tid

        return ids
