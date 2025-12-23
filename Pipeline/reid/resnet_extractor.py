#!/usr/bin/env python3
"""
Torchreid-based ReID Extractor (ResNet backbone)
------------------------------------------------
Uses Torchreid's official model builder for compatibility
with Market1501/MSMT17 checkpoints (e.g., resnet50_market_xent.pth.tar).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from PIL import Image
import torchreid

# ------------------------------------------------------------------
# PyTorch 2.6 safety: allow Torchreid's older numpy scalar in checkpoints
# ------------------------------------------------------------------
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    # Older PyTorch may not have this API; ignore in that case.
    pass


class ResNetExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet50",
                 device: str = "cuda",
                 weights: str | None = None):
        super().__init__()
        self.device = torch.device(device)

        # ---------------- Build model ----------------
        print(f"[INFO] Building Torchreid model: {backbone}")
        self.model = torchreid.models.build_model(
            name=backbone,
            num_classes=751,   # Market1501 classes
            pretrained=True    # load pretrained ImageNet weights first
        )

        # ---------------- Load pretrained ReID weights ----------------
        if weights and weights not in ["imagenet", None]:
            weights_path = Path(weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"ReID weights not found: {weights_path}")
            print(f"[INFO] Loading Torchreid weights from {weights_path}")

            try:
                # Explicitly disable weights_only restriction (trusted checkpoint)
                try:
                    checkpoint = torch.load(
                        weights_path,
                        map_location=self.device,
                        weights_only=False,   # PyTorch 2.6+
                    )
                except TypeError:
                    # Older PyTorch: no weights_only argument
                    checkpoint = torch.load(
                        weights_path,
                        map_location=self.device,
                    )
                except UnicodeDecodeError:
                    print("[WARN] UnicodeDecodeError; retrying with latin1 encoding…")
                    try:
                        checkpoint = torch.load(
                            weights_path,
                            map_location=self.device,
                            encoding="latin1",
                            weights_only=False,
                        )
                    except TypeError:
                        checkpoint = torch.load(
                            weights_path,
                            map_location=self.device,
                            encoding="latin1",
                        )

                # Torchreid checkpoints usually store weights under "state_dict"
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Clean possible prefixes
                cleaned = {
                    k.replace("module.", "").replace("backbone.", "").replace("base.", ""): v
                    for k, v in state_dict.items()
                }

                missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
                print("[INFO] Torchreid ResNet weights loaded successfully.")
                if missing or unexpected:
                    print(f"[DEBUG] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

            except Exception as e:
                print(f"[WARN] Could not fully load Torchreid ResNet weights: {e}")

        self.model.eval().to(self.device)
        self.feat_dim = self.model.feature_dim if hasattr(self.model, "feature_dim") else 2048

        # ---------------- Preprocessing transform ----------------
        self.transform = T.Compose([
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # -------------------------------------------------------------
    @torch.no_grad()
    def forward(self, crops):
        """
        crops: list of np.ndarray (BGR) or PIL.Image
        Returns: torch.Tensor (N, feat_dim) L2-normalized embeddings.
        """
        if not crops:
            return torch.empty((0, self.feat_dim), device=self.device)

        tensors = []
        for c in crops:
            if isinstance(c, np.ndarray):
                # OpenCV BGR -> PIL RGB
                c = Image.fromarray(c[:, :, ::-1].astype(np.uint8))
            if not isinstance(c, torch.Tensor):
                c = self.transform(c)
            tensors.append(c)

        if not tensors:
            return torch.empty((0, self.feat_dim), device=self.device)

        x = torch.stack(tensors).to(self.device)
        feats = self.model(x)
        if isinstance(feats, tuple):
            feats = feats[0]  # some Torchreid models return (feat, logits)
        feats = F.normalize(feats, p=2, dim=1)
        return feats


# -------------------------------------------------------
def build_reid(cfg):
    """Factory hook for other pipelines that use dict cfg."""
    if isinstance(cfg, dict):
        return ResNetExtractor(
            backbone=cfg.get("backbone", "resnet50"),
            device=cfg.get("device", "cuda"),
            weights=cfg.get("weights", None),
        )
    else:
        return ResNetExtractor(backbone="resnet50", device="cuda", weights=None)
