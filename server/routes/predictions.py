import json
import os
from fastapi import APIRouter
from server.services import data_cache

router = APIRouter()

GEOJSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "data", "predictions.geojson"
)


@router.get("/predictions")
async def get_predictions():
    cached = data_cache.get("predictions")
    if cached:
        return cached

    path = os.path.abspath(GEOJSON_PATH)
    if not os.path.exists(path):
        return {"features": []}

    with open(path) as f:
        gj = json.load(f)

    result = {"features": gj.get("features", [])}
    data_cache.set("predictions", result)
    return result
