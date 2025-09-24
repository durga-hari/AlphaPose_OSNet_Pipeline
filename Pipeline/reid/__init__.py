from __future__ import annotations
from typing import Any
from .osnet_extractor import OSNetExtractor
from common import Registry


REID = Registry()
REID.register("osnet")(OSNetExtractor)
REID.register("none")(lambda **_: None)


def build_reid(name: str, **kwargs) -> Any:
    return REID.build(name, **kwargs)