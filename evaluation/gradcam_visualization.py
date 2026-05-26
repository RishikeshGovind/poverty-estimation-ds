import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

cfg = load_config()
norm_factor = cfg["sentinel2"]["normalization_factor"]
checkpoint_path = cfg["evaluation"]["checkpoint"]

# Accept .npy (raw Sentinel-2) or .jpg/.png (treated as 8-bit, will be rescaled)
image_path = "example_satellite.npy"

if image_path.endswith(".npy"):
    img_raw = np.load(image_path).astype(np.float32)          # (C, H, W)
    img_norm = np.clip(img_raw / norm_factor, 0, 1)           # [0, 1] matching training
    img_np = np.transpose(img_norm[:3], (1, 2, 0))            # (H, W, 3) for visualization
else:
    logger.warning(
        "Loading JPEG/PNG input. Pixel statistics differ from training (trained on S2 /10000). "
        "Use a .npy Sentinel-2 patch for accurate results."
    )
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np = np.array(img).astype(np.float32) / 255.0         # (H, W, 3)
    img_norm = np.transpose(img_np, (2, 0, 1))                # (C, H, W)

input_tensor = torch.from_numpy(
    img_norm if img_norm.ndim == 3 and img_norm.shape[0] <= 4
    else np.transpose(img_np, (2, 0, 1))
).unsqueeze(0)

model = ResNetRegression(in_channels=3)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()
logger.info("Loaded model from %s", checkpoint_path)

target_layers = [model.model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
gradcam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]

visualization = show_cam_on_image(img_np[:, :, :3], gradcam_map, use_rgb=True)

output_path = "outputs/gradcam_overlay.png"
Image.fromarray(visualization).save(output_path)
logger.info("Grad-CAM overlay saved to %s", output_path)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_np[:, :, :3])
plt.title("Input Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Overlay")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/gradcam_comparison.png")
