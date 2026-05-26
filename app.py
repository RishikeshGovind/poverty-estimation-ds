import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)
cfg = load_config()
norm_factor = cfg["sentinel2"]["normalization_factor"]
checkpoint_path = cfg["training"]["checkpoint"]

st.title("Poverty Prediction from Satellite Image")

uploaded_file = st.file_uploader(
    "Upload a Sentinel-2 patch (.npy) or preview image (.jpg/.png)",
    type=["npy", "jpg", "jpeg", "png"],
)

if uploaded_file:
    name = uploaded_file.name

    if name.endswith(".npy"):
        img_raw = np.load(uploaded_file).astype(np.float32)       # (C, H, W)
        img_norm = np.clip(img_raw / norm_factor, 0, 1)           # matches training
        img_display = np.transpose(img_norm[:3], (1, 2, 0))       # (H, W, 3) for display
        input_tensor = torch.from_numpy(img_norm[:3]).unsqueeze(0)
    else:
        st.warning(
            "JPEG/PNG inputs have different pixel statistics than the training data "
            "(model trained on Sentinel-2 values /10000). Upload a .npy patch for accurate results."
        )
        image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        img_display = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
        img_norm = np.transpose(img_display, (2, 0, 1))
        input_tensor = torch.from_numpy(img_norm).unsqueeze(0)

    st.image(img_display, caption="Uploaded Image", use_column_width=True)

    model = ResNetRegression(in_channels=3)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    except FileNotFoundError:
        st.error(f"Model checkpoint not found at {checkpoint_path}. Train the model first.")
        st.stop()
    model.eval()

    with torch.no_grad():
        pred = model(input_tensor).item()

    st.subheader(f"Predicted Poverty Score: {pred:.2f}")
    logger.info("Inference result: %.4f for file %s", pred, name)

    fig, ax = plt.subplots()
    ax.imshow(img_display[:, :, :3])
    ax.set_title("Input Image")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Upload a Sentinel-2 .npy patch to get a prediction.")
