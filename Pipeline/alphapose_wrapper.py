# Pipeline/alphapose_wrapper.py
import os, sys, types, traceback
from importlib import import_module
import yaml
import torch


try:
    from easydict import EasyDict as _ED  # noqa
except ModuleNotFoundError:
    import types as _t
    class _EasyDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    _shim = _t.ModuleType("easydict"); _shim.EasyDict = _EasyDict
    sys.modules["easydict"] = _shim
# ------------------------------------------------------------------------------

def _ensure_repo_on_path(alpha_base: str):
    alpha_base = os.path.abspath(alpha_base)
    if not os.path.isdir(alpha_base):
        raise FileNotFoundError(f"alphapose_path is not a directory: {alpha_base}")
    if alpha_base not in sys.path:
        sys.path.insert(0, alpha_base)
    if not os.path.isdir(os.path.join(alpha_base, "alphapose")):
        raise ModuleNotFoundError(
            f"'alphapose/' not found under {alpha_base}. "
            "Point alphapose_path to the repo root that contains the 'alphapose' folder."
        )

def _choose_device(req=None):
    if req and req.startswith("cuda") and torch.cuda.is_available():
        return req
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _load_yaml(path: str):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_alphapose(config):
    """
    Expects in config:
      alphapose_path, alphapose_config, alphapose_checkpoint,
      detector_checkpoint, [detector_config], [device], [det_* thresholds]
    Returns: detector, pose_model, run_pose (callable placeholder)
    """
    try:
        # 1) import paths
        _ensure_repo_on_path(getattr(config, "alphapose_path"))

        # 2) imports for *your* fork layout
        pose_builder = import_module("alphapose.models.builder")
        get_detector = getattr(import_module("detector.apis"), "get_detector")

        # 3) device setup
        device = _choose_device(getattr(config, "device", None))
        torch.backends.cudnn.benchmark = bool(getattr(config, "cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = bool(getattr(config, "cudnn_deterministic", False))
        print(f"[AP] Device: {device}")

        # 4) detector
        model_cfg = getattr(config, "detector_config", None)
        model_weights = getattr(config, "detector_checkpoint", None)
        if not model_weights:
            raise FileNotFoundError("detector_checkpoint must be provided")

        gpu_idx = 0 if device.startswith("cuda") else -1
        det_cfg = types.SimpleNamespace(
            detector="yolo",
            gpus=[gpu_idx],
            detbatch=1,
            nms=float(getattr(config, "det_nms_threshold", 0.45)),
            confidence=float(getattr(config, "det_conf_threshold", 0.25)),
            device=device,
            # expose both naming styles for cross-fork compatibility
            model_cfg=os.path.abspath(model_cfg) if model_cfg else None,
            model_weights=os.path.abspath(model_weights),
            cfgfile=os.path.abspath(model_cfg) if model_cfg else None,
            weightfile=os.path.abspath(model_weights),
        )
        detector = get_detector(det_cfg)

        # 5) pose model
        pose_cfg = _load_yaml(getattr(config, "alphapose_config"))
        ckpt = getattr(config, "alphapose_checkpoint", None)
        if not ckpt:
            raise FileNotFoundError("alphapose_checkpoint must be provided")

        build_sppe = getattr(pose_builder, "build_sppe", None)
        if build_sppe is None and hasattr(pose_builder, "builder"):
            build_sppe = getattr(pose_builder.builder, "build_sppe", None)
        if build_sppe is None:
            raise AttributeError("Couldn't find 'build_sppe' in alphapose.models.builder")

        pose_model = build_sppe(pose_cfg, ckpt, device=device)

        if getattr(config, "use_fp16", False) and device.startswith("cuda"):
            try:
                pose_model.half()
                print("[AP] Pose model FP16 enabled.")
            except Exception:
                print("[AP] Warning: FP16 requested but not applied.")

        # 6) minimal placeholder; replace with your actual inference if your pipeline calls it
        def run_pose(*args, **kwargs):
            raise NotImplementedError("run_pose is a placeholder—pipeline should handle inference stages.")

        return detector, pose_model, run_pose

    except Exception as e:
        traceback.print_exc()
        raise ImportError(f"AlphaPose init failed: {e}") from e
