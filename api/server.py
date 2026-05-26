import io
import json
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from models.resnet_model import ResNetRegression
from scoring.sdg_scorer import SDGScorer
from utils.config import load_config
from utils.logging import get_logger
from utils.uncertainty import mc_predict

logger = get_logger(__name__)
cfg = load_config()
norm_factor   = cfg["sentinel2"]["normalization_factor"]
checkpoint_path = cfg["training"]["checkpoint"]
api_cfg       = cfg["api"]
geojson_path  = api_cfg.get("geojson_path", "docs/data/predictions.geojson")

app = FastAPI(title="Poverty Estimation API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_cfg.get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = SDGScorer()

try:
    _model = ResNetRegression(
        in_channels=3,
        dropout_p=cfg.get("model", {}).get("dropout_p", 0.3),
    )
    _model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    _model.eval()
    logger.info("Model loaded from %s", checkpoint_path)
except FileNotFoundError:
    logger.error("Checkpoint not found at %s. Train the model first.", checkpoint_path)
    _model = None


def _preprocess(content: bytes, filename: str) -> torch.Tensor:
    if filename.endswith(".npy"):
        arr = np.load(io.BytesIO(content)).astype(np.float32)
        return torch.from_numpy(np.clip(arr[:3] / norm_factor, 0, 1)).unsqueeze(0)
    elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
        logger.warning("JPEG/PNG upload — pixel statistics differ from training data.")
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((256, 256))
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
    else:
        raise HTTPException(400, "Unsupported format. Upload .npy, .jpg, or .png.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Train the model first.")

    content = await file.read()
    tensor = _preprocess(content, file.filename or "")

    mean, std = mc_predict(_model, tensor, device=torch.device("cpu"))
    wealth = float(mean[0])
    uncertainty = float(std[0])

    scores = scorer.score_row({"prediction": wealth})
    logger.info("Prediction: %.4f ± %.4f for %s", wealth, uncertainty, file.filename)

    return JSONResponse({
        "wealth_index": round(wealth, 4),
        "uncertainty":  round(uncertainty, 4),
        **scores,
    })


@app.get("/scores")
def get_scores():
    """Return pre-computed GeoJSON for the frontend map."""
    if not os.path.exists(geojson_path):
        raise HTTPException(
            404,
            f"GeoJSON not found at {geojson_path}. Run: python -m scoring.generate_geojson",
        )
    with open(geojson_path) as f:
        return JSONResponse(json.load(f))


@app.get("/sdg-summary")
def sdg_summary():
    """Aggregate SDG scores by country."""
    if not os.path.exists(geojson_path):
        raise HTTPException(404, "GeoJSON not found. Run scoring.generate_geojson first.")

    with open(geojson_path) as f:
        geojson = json.load(f)

    by_country: dict = {}
    for feat in geojson["features"]:
        p = feat["properties"]
        country = p.get("country", "Unknown")
        if country not in by_country:
            by_country[country] = {"sdg1": [], "sdg7": [], "sdg11": [], "composite": []}
        by_country[country]["sdg1"].append(p.get("sdg1_score"))
        by_country[country]["sdg7"].append(p.get("sdg7_score"))
        by_country[country]["sdg11"].append(p.get("sdg11_score"))
        by_country[country]["composite"].append(p.get("composite_score"))

    def _mean(lst):
        vals = [v for v in lst if v is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    summary = {
        country: {
            "n_clusters":      len(vals["composite"]),
            "sdg1_mean":       _mean(vals["sdg1"]),
            "sdg7_mean":       _mean(vals["sdg7"]),
            "sdg11_mean":      _mean(vals["sdg11"]),
            "composite_mean":  _mean(vals["composite"]),
        }
        for country, vals in by_country.items()
    }
    return JSONResponse(summary)


if __name__ == "__main__":
    uvicorn.run("api.server:app", host=api_cfg["host"], port=api_cfg["port"], reload=True)
