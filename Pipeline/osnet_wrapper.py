# Pipeline/osnet_wrapper.py
import torch
from torchvision import transforms as T

# torchreid imports
import torchreid
from torchreid import models

def _build_osnet(model_name: str = "osnet_x1_0", device: str = "cuda"):
    # num_classes dummy since we just do feature extraction
    model = models.build_model(
        name=model_name,
        num_classes=1000,
        loss="softmax",
        pretrained=False,   # we'll load our own checkpoint
    )
    model.eval().to(device)
    return model

def _safe_load_state_dict(path: str):
    """
    Robust loader compatible with PyTorch >=2.6 where weights_only=True is default.
    Forces weights_only=False and unwraps common state_dict layouts.
    """
    # IMPORTANT: weights_only=False to allow legacy pickled checkpoints
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Common layouts
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format at {path}: {type(ckpt)}")

    # Strip possible 'module.' prefixes
    state = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        state[nk] = v
    return state

def _make_transform(size=256):
    # Basic, deterministic transform for person ReID crops
    return T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def load_osnet(config):
    """
    Returns:
      - reid_model: torch.nn.Module on config.device
      - reid_tf: callable(np.ndarray HxWxC BGR or RGB) -> torch.FloatTensor CHW (RGB normalized)
    """
    device = config.device if hasattr(config, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = getattr(config, "osnet_model", "osnet_x1_0")
    weights = getattr(config, "osnet_weights", "")

    print(f"[INFO] Loading OSNet weights: {weights}")

    # Build model
    model = _build_osnet(model_name=model_name, device=device)

    # Load checkpoint robustly (bypass torchreid helper to force weights_only=False)
    state = _safe_load_state_dict(weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[OSNET] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[OSNET] Unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")

    # Simple transform for BGR/RGB numpy image crops from pipeline._extract_feats
    tf = _make_transform()

    def reid_tf(img_bgr_or_rgb):
        import numpy as np, cv2
        arr = img_bgr_or_rgb
        if isinstance(arr, torch.Tensor):
            # assume already CHW float [0,1]
            x = arr
        else:
            # numpy HxWxC (BGR or RGB). Ensure RGB uint8
            if arr is None or arr.size == 0:
                # return zero tensor to keep pipeline flowing
                return torch.zeros(3, 256, 256, dtype=torch.float32)
            if arr.shape[-1] == 3:
                # assume BGR from OpenCV, convert to RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            x = tf(arr)  # CHW float
        return x

    return model, reid_tf
