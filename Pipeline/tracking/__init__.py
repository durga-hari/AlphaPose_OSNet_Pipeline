from __future__ import annotations
from typing import Any

from common import Registry
from .strongsort_tracker import StrongSORTTracker
from .deepsort_tracker import DeepSortTracker

TRACKERS = Registry()

# register both trackers
TRACKERS.register("strongsort")(StrongSORTTracker)
TRACKERS.register("deepsort")(DeepSortTracker)


def build_tracker(name: str, **kwargs) -> Any:
    """
    Factory used by ap_pipeline.SequentialPipeline.

    name:
      - "strongsort" -> StrongSORTTracker
      - "deepsort"   -> DeepSortTracker
      - "none" / ""  -> no tracker
    """
    if name is None:
        return None
    name = str(name).lower().strip()
    if name in {"none", "", "off"}:
        return None
    return TRACKERS.build(name, **kwargs)
