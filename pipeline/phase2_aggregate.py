"""
Phase 2 Task 5 — Join country-level satellite features to DHS cluster training data.

Reads:
  data/training_dataset.csv          — 3048 DHS clusters (Kenya + Nigeria)
  pipeline/outputs/satellite_features.json — country-level NTL / NDVI / NDBI

Writes:
  pipeline/outputs/training_with_satellite.csv
  (same rows, extra columns: ntl_2022, ntl_2023, ntl_trend, ndvi_2023, ndbi_2023)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PIPELINE   = Path(__file__).parent
SAT_JSON   = PIPELINE / "outputs" / "satellite_features.json"
TRAIN_CSV  = Path("data/training_dataset.csv")
OUT_CSV    = PIPELINE / "outputs" / "training_with_satellite.csv"


def ntl_trend(ntl_by_year: dict) -> float:
    """Linear slope of NTL radiance over available years (nW/cm²/sr per year)."""
    years = sorted(int(y) for y in ntl_by_year)
    if len(years) < 2:
        return 0.0
    vals = [ntl_by_year[str(y)] for y in years]
    slope = np.polyfit(years, vals, 1)[0]
    return round(float(slope), 6)


def main():
    print("[aggregate] Loading data…")
    df  = pd.read_csv(TRAIN_CSV)
    sat = json.load(open(SAT_JSON))

    # Build a lookup: country_lower → feature dict
    lookup: dict[str, dict] = {}
    for iso3, feats in sat.items():
        # Match by country name in training CSV ("Kenya", "Nigeria")
        # We need to map ISO3 → country name used in training data
        pass

    # ISO3 → name as used in training_dataset.csv
    ISO3_TO_TRAIN_NAME = {
        "KEN": "Kenya",
        "NGA": "Nigeria",
    }
    TRAIN_NAME_TO_ISO3 = {v: k for k, v in ISO3_TO_TRAIN_NAME.items()}

    def get_feature(country_name: str, feat: str, year: str, default=0.0):
        iso3 = TRAIN_NAME_TO_ISO3.get(country_name)
        if not iso3 or iso3 not in sat:
            return default
        return sat[iso3].get(feat, {}).get(year, default)

    def get_ntl_trend(country_name: str) -> float:
        iso3 = TRAIN_NAME_TO_ISO3.get(country_name)
        if not iso3 or iso3 not in sat:
            return 0.0
        return ntl_trend(sat[iso3].get("ntl", {}))

    print("[aggregate] Joining satellite features…")
    df["ntl_2022"]   = df["country"].map(lambda c: get_feature(c, "ntl",  "2022"))
    df["ntl_2023"]   = df["country"].map(lambda c: get_feature(c, "ntl",  "2023"))
    df["ntl_trend"]  = df["country"].map(get_ntl_trend)
    df["ndvi_2023"]  = df["country"].map(lambda c: get_feature(c, "ndvi", "2023"))
    df["ndbi_2023"]  = df["country"].map(lambda c: get_feature(c, "ndbi", "2023"))
    df["is_urban"]   = (df["URBAN_RURA"] == "U").astype(int)

    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[aggregate] Saved {len(df)} rows → {OUT_CSV}")
    print(f"[aggregate] New columns: ntl_2022, ntl_2023, ntl_trend, ndvi_2023, ndbi_2023, is_urban")
    print(f"[aggregate] Country breakdown:")
    print(df.groupby("country")[["ntl_2023", "ndvi_2023", "ndbi_2023"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
