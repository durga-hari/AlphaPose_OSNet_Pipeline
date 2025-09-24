from __future__ import annotations
from typing import Any
from .alphapose_estimator import AlphaPoseEstimator,AlphaPoseCfg
from common import Registry


POSE_ESTIMATORS = Registry()
POSE_ESTIMATORS.register("alphapose")(AlphaPoseEstimator)
# POSE_ESTIMATORS.register("alphapose")(AlphaPoseCfg)

def build_pose_estimator(name: str, **kwargs) -> Any:
    return POSE_ESTIMATORS.build(name, **kwargs)

# Make AlphaPoseCfg available at package level
__all__ = ["build_pose_estimator", "AlphaPoseEstimator", "AlphaPoseCfg"]