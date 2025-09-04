# pipeline/osnet_wrapper.py
import os, torch, torchvision.transforms as transforms
import torchreid

def load_osnet(config):
    torch.backends.cudnn.benchmark = bool(getattr(config, "cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(getattr(config, "cudnn_deterministic", False))

    model = torchreid.models.build_model(
        name=config.osnet_model, num_classes=1041, loss="softmax", pretrained=False
    )
    if not os.path.isfile(config.osnet_weights):
        raise FileNotFoundError(f"OSNet weights not found: {config.osnet_weights}")
    print(f"[INFO] Loading OSNet weights: {config.osnet_weights}")
    torchreid.utils.load_pretrained_weights(model, config.osnet_weights)

    device = torch.device(config.device if isinstance(config.device, str) else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    if getattr(config, "use_fp16", False) and device.type == "cuda":
        try:
            model.half()
            print("[INFO] OSNet set to FP16.")
        except Exception:
            print("[WARN] FP16 requested but not applied to OSNet.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform
