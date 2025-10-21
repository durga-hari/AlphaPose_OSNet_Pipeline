# Pipeline/pose/hand_estimator.py
import numpy as np
import torch

class HandEstimator:
    def __init__(self, cfg_yaml, ckpt, device="cuda"):
        from alphapose.models.builder import build_sppe
        from alphapose.utils.config import update_config
        import yaml

        with open(cfg_yaml) as f:
            ap_cfg = yaml.safe_load(f)

        self.device = device
        self.model = build_sppe(ap_cfg['MODEL'], ap_cfg['DATA_PRESET'])
        self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model = self.model.to(device).eval()

    def __call__(self, img_crops):
        """
        img_crops: list of cropped BGR hand images
        returns: list of (21,3) keypoints per hand
        """
        results = []
        with torch.no_grad():
            for crop in img_crops:
                inp = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float().to(self.device)
                out = self.model(inp)
                kpts = out.cpu().numpy().squeeze()
                results.append(kpts)
        return results
