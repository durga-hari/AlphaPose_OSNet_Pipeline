from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import os
import numpy as np




def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)




def safe_stem(p: Path | str) -> str:
    try:
        return Path(p).stem
    except Exception:
        s = str(p)
        return os.path.splitext(os.path.basename(s))[0]

# Simple plugin registry to make swapping components easy
class Registry:
    def __init__(self):
        self._items: Dict[str, Any] = {}

    def register(self, name: str):
        def deco(cls):
            key = name.lower()
            self._items[key] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs) -> Any:
        key = (name or "").lower()
        if key not in self._items:
            raise KeyError(f"Unknown component '{name}'. Available: {list(self._items)}")
        return self._items[key](**kwargs)

# Shared dataclasses
@dataclass
class DetResult:
    boxes_xyxy: "np.ndarray" # (N,4)
    scores: "np.ndarray | None" = None
    labels: "np.ndarray | None" = None

@dataclass
class PoseResult:
    keypoints_list: list # list[(K,3) as [x,y,conf]] per detection
    pose_scores: list # list[float]

