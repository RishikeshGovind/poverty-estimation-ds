"""
GET /api/satellite/{iso3}

Returns the raw satellite feature time series for one country from
pipeline/outputs/satellite_features.json. Used by the dashboard to show
actual VIIRS radiance, MODIS NDVI, and Landsat NDBI in the country popup.
"""

import json
import os
import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter()

_SAT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "pipeline", "outputs", "satellite_features.json"
)

_SAT_CACHE: dict | None = None


def _load() -> dict:
    global _SAT_CACHE
    if _SAT_CACHE is None:
        if not os.path.exists(_SAT_PATH):
            _SAT_CACHE = {}
        else:
            with open(_SAT_PATH) as f:
                _SAT_CACHE = json.load(f)
    return _SAT_CACHE


def _ntl_trend(ntl: dict) -> float:
    years = sorted(int(y) for y in ntl)
    if len(years) < 2:
        return 0.0
    vals = [ntl[str(y)] for y in years]
    return round(float(np.polyfit(years, vals, 1)[0]), 6)


def _interpret_ntl(val: float, trend: float) -> str:
    direction = "rising" if trend > 0.005 else "falling" if trend < -0.005 else "stable"
    level = "high" if val > 0.5 else "moderate" if val > 0.25 else "low"
    return f"{level.capitalize()} electrification, {direction}"


def _interpret_ndvi(val: float) -> str:
    if val > 0.45:
        return "Dense vegetation / strong food security signal"
    if val > 0.3:
        return "Moderate vegetation"
    if val > 0.15:
        return "Sparse vegetation / semi-arid"
    return "Very low vegetation / arid"


def _interpret_ndbi(val: float) -> str:
    if val > 0.1:
        return "High urban / built-up density"
    if val > 0.03:
        return "Moderate built-up areas"
    return "Low urban density / mostly rural"


@router.get("/satellite/{iso3}")
async def get_satellite(iso3: str):
    sat = _load()
    iso3 = iso3.upper()
    if iso3 not in sat:
        raise HTTPException(status_code=404, detail=f"No satellite data for {iso3}")

    data = sat[iso3]
    ntl  = data.get("ntl",  {})
    ndvi = data.get("ndvi", {})
    ndbi = data.get("ndbi", {})

    trend = _ntl_trend(ntl)
    ntl_latest  = float(ntl.get("2023",  ntl.get("2022",  0)))
    ndvi_latest = float(ndvi.get("2023", ndvi.get("2022", 0)))
    ndbi_latest = float(ndbi.get("2023", ndbi.get("2022", 0)))

    return {
        "iso3": iso3,
        "ntl": {
            "values":        {y: round(v, 4) for y, v in ntl.items()},
            "latest":        round(ntl_latest, 4),
            "trend_per_year": round(trend, 6),
            "trend_label":   f"{'+' if trend >= 0 else ''}{trend:.4f} nW/cm²/sr per year",
            "source":        "VIIRS Black Marble (NASA)",
            "interpretation": _interpret_ntl(ntl_latest, trend),
        },
        "ndvi": {
            "values":        {y: round(v, 4) for y, v in ndvi.items()},
            "latest":        round(ndvi_latest, 4),
            "source":        "MODIS MOD13A3 (NASA GSFC)",
            "interpretation": _interpret_ndvi(ndvi_latest),
        },
        "ndbi": {
            "values":        {y: round(v, 4) for y, v in ndbi.items()},
            "latest":        round(ndbi_latest, 4),
            "source":        "Landsat 8/9 OLI (USGS)",
            "interpretation": _interpret_ndbi(ndbi_latest),
        },
    }
