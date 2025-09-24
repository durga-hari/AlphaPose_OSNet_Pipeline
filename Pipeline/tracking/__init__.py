from __future__ import annotations
from typing import Any
from common import Registry
from .strongsort_tracker import StrongSORTTracker


TRACKERS = Registry()


# Optional: register external StrongSORT wrapper here later.
TRACKERS.register("strongsort")(StrongSORTTracker)

def build_tracker(name: str, **kwargs) -> Any:
    return TRACKERS.build(name, **kwargs)