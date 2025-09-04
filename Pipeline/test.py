import sys
import os
import torch

# Ensure the local pipeline directory is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import Config
from alphapose_wrapper import load_alphapose

def test_alphapose(config):
    print("[INFO] Testing AlphaPose integration...")
    try:
        detector, pose_model, run_inference = load_alphapose(config)
        print("[SUCCESS] AlphaPose models loaded.")
        print(f"  Detector:     {type(detector)}")
        print(f"  Pose model:   {type(pose_model)}")
    except Exception as e:
        print(f"[ERROR] AlphaPose failed to load:\n{e}")

def test_osnet(config):
    print("\n[INFO] Testing OSNet (torchreid) integration...")
    try:
        import torchreid

        model = torchreid.models.build_model(
            name=config.osnet_model,
            num_classes=1000  # dummy
        )
        torchreid.utils.load_pretrained_weights(model, config.osnet_weights)
        model = model.to(config.device).eval()
        print("[SUCCESS] OSNet model loaded.")
        print(f"  Model: {config.osnet_model}")
        print(f"  Device: {config.device}")
    except Exception as e:
        print(f"[ERROR] OSNet failed to load:\n{e}")

if __name__ == "__main__":
    config = Config()
    test_alphapose(config)
    test_osnet(config)
