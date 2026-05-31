"""
Phase 2 Task 5 — Extract cluster-level satellite features from downloaded patches.

Previous version joined country-level averages → all Kenya clusters got the same
NTL value → satellite features had near-zero importance vs lat/lon.

This version reads each cluster's own 64×64 patch to get real spatial variation:

  VIIRS patch  (1×64×64 float32) → ntl_mean, ntl_std, ntl_max
  S2 patch     (3×64×64 float32, RGB B04/B03/B02) →
                 s2_red, s2_green, s2_blue,
                 s2_exgreen  (Excess Green = 2G−R−B, vegetation proxy w/o NIR),
                 s2_brightness ((R+G+B)/3, urban/reflectance proxy)

  NTL trend from satellite_features.json (country-level slope 2019→2023)
  is_urban from DHS survey flag

Reads:
  data/training_dataset.csv
  data/patches/viirs_*/viirs_NNNNN.npy   (via viirs_patch_file column)
  data/patches/s2_*/s2_NNNNN.npy         (via s2_patch_file column)
  pipeline/outputs/satellite_features.json

Writes:
  pipeline/outputs/training_with_satellite.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT       = Path(__file__).parent.parent
PIPELINE   = Path(__file__).parent
SAT_JSON   = PIPELINE / "outputs" / "satellite_features.json"
TRAIN_CSV  = ROOT / "data" / "training_dataset.csv"
OUT_CSV    = PIPELINE / "outputs" / "training_with_satellite.csv"

ISO3_BY_COUNTRY = {"Kenya": "KEN", "Nigeria": "NGA"}


def ntl_trend(ntl_by_year: dict) -> float:
    years = sorted(int(y) for y in ntl_by_year)
    if len(years) < 2:
        return 0.0
    vals = [ntl_by_year[str(y)] for y in years]
    return round(float(np.polyfit(years, vals, 1)[0]), 6)


def extract_viirs(path: str) -> dict:
    arr = np.load(ROOT / path).astype(np.float32).ravel()
    return {
        "ntl_mean": float(arr.mean()),
        "ntl_std":  float(arr.std()),
        "ntl_max":  float(arr.max()),
    }


def extract_s2(path: str) -> dict:
    arr = np.load(ROOT / path).astype(np.float32)   # (3, 64, 64)
    r, g, b = arr[0].mean(), arr[1].mean(), arr[2].mean()
    exgreen = float(2 * g - r - b)
    return {
        "s2_red":        float(r),
        "s2_green":      float(g),
        "s2_blue":       float(b),
        "s2_exgreen":    round(exgreen, 6),
        "s2_brightness": float((r + g + b) / 3),
    }


def main():
    print("[aggregate] Loading training data and satellite features…")
    df  = pd.read_csv(TRAIN_CSV)
    sat = json.load(open(SAT_JSON))

    rows = []
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patches"):
        feat: dict = {}

        # Cluster-level VIIRS features
        try:
            feat.update(extract_viirs(row["viirs_patch_file"]))
        except Exception:
            feat.update({"ntl_mean": np.nan, "ntl_std": np.nan, "ntl_max": np.nan})
            errors += 1

        # Cluster-level S2 features
        try:
            feat.update(extract_s2(row["s2_patch_file"]))
        except Exception:
            feat.update({"s2_red": np.nan, "s2_green": np.nan, "s2_blue": np.nan,
                         "s2_exgreen": np.nan, "s2_brightness": np.nan})
            errors += 1

        # Country-level NTL trend (temporal trajectory — still useful at country grain)
        iso3 = ISO3_BY_COUNTRY.get(row["country"], "")
        feat["ntl_trend"] = ntl_trend(sat.get(iso3, {}).get("ntl", {})) if iso3 in sat else 0.0

        feat["is_urban"]   = 1 if row["URBAN_RURA"] == "U" else 0
        feat["latitude"]   = row["latitude"]
        feat["longitude"]  = row["longitude"]
        feat["wealth_index"] = row["wealth_index"]
        feat["country"]    = row["country"]
        feat["ADM1NAME"]   = row["ADM1NAME"]
        rows.append(feat)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n[aggregate] Saved {len(out)} rows → {OUT_CSV}")
    if errors:
        print(f"[aggregate] {errors} patch load errors (rows kept with NaN, dropped at train time)")

    print("\n[aggregate] Feature means by country:")
    feat_cols = ["ntl_mean", "ntl_std", "ntl_max", "s2_exgreen", "s2_brightness", "ntl_trend"]
    print(out.groupby("country")[feat_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
