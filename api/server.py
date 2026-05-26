from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
import io
from PIL import Image

from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)
cfg = load_config()
norm_factor = cfg["sentinel2"]["normalization_factor"]
checkpoint_path = cfg["training"]["checkpoint"]
api_cfg = cfg["api"]

app = FastAPI()

try:
    model = ResNetRegression(in_channels=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    logger.info("Model loaded from %s", checkpoint_path)
except FileNotFoundError:
    logger.error("Checkpoint not found at %s. Train the model before starting the server.", checkpoint_path)
    model = None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    content = await file.read()
    name = file.filename or ""

    if name.endswith(".npy"):
        img_raw = np.load(io.BytesIO(content)).astype(np.float32)  # (C, H, W)
        img_norm = np.clip(img_raw / norm_factor, 0, 1)
        input_tensor = torch.from_numpy(img_norm[:3]).unsqueeze(0)
    elif name.lower().endswith((".jpg", ".jpeg", ".png")):
        logger.warning("JPEG/PNG upload from %s — statistics differ from training data.", file.filename)
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((256, 256))
        img_np = np.array(image).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload .npy, .jpg, or .png.")

    with torch.no_grad():
        pred = model(input_tensor).item()

    logger.info("Prediction: %.4f for file %s", pred, file.filename)
    return JSONResponse({"prediction": pred})


if __name__ == "__main__":
    uvicorn.run("api.server:app", host=api_cfg["host"], port=api_cfg["port"], reload=True)
