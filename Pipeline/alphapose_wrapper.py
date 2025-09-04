# pipeline/alphapose_wrapper.py
import sys, os, traceback, types, yaml, torch

def choose_device(requested_device=None):
    if requested_device and requested_device.startswith("cuda") and torch.cuda.is_available():
        return requested_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def load_yaml_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "TYPE" not in cfg:
        raise ValueError("Invalid AlphaPose YAML (missing TYPE)")
    return cfg

def setup_sys_path(alpha_base):
    alpha_pkg = os.path.join(alpha_base, "alphapose")
    sys.path.insert(0, alpha_base)
    sys.path.insert(0, alpha_pkg)

def load_alphapose(config):
    try:
        alpha_base = os.path.abspath(config.alphapose_path)
        setup_sys_path(alpha_base)

        from AlphaPose.alphapose.models import builder as pose_builder
        from AlphaPose.detector.apis import get_detector

        device = choose_device(getattr(config, "device", None))
        torch.backends.cudnn.benchmark = bool(getattr(config, "cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = bool(getattr(config, "cudnn_deterministic", False))
        print(f"[INFO] Running models on device: {device}")

        # Detector (YOLO)
        model_cfg = os.path.abspath(config.detector_config)
        model_weights = os.path.abspath(config.detector_checkpoint)

        gpu_index = 0 if device.startswith("cuda") else -1
        det_cfg = types.SimpleNamespace(
            detector="yolo",
            gpus=[gpu_index],
            detbatch=1,
            nms=float(getattr(config, "det_nms_threshold", 0.45)),
            confidence=float(getattr(config, "det_conf_threshold", 0.25)),
            device=device,
            model_cfg=model_cfg,
            model_weights=model_weights,
            cfgfile=model_cfg,
            weightfile=model_weights,
        )
        detector = get_detector(det_cfg)

        # Pose model
        pose_cfg = load_yaml_config(config.alphapose_config)
        ckpt = config.alphapose_checkpoint
        pose_model = pose_builder.build_sppe(pose_cfg, ckpt, device=device)
        if getattr(config, "use_fp16", False) and device.startswith("cuda"):
            try:
                pose_model.half()
                print("[INFO] Pose model set to FP16.")
            except Exception:
                print("[WARN] FP16 requested but not applied to pose model.")
        return detector, pose_model
    except Exception as e:
        traceback.print_exc()
        raise ImportError(f"AlphaPose import failed: {e}")
