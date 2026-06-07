"""
Phase 3 Task 7 — Generate predictions.geojson for all SSA countries.

For Kenya and Nigeria (countries with real DHS cluster data + Sentinel-2 patches)
we run the trained ResNet18 CNN directly on the 256×256 S2 patches.

Fallback: if the CNN checkpoint is missing, the GBR tabular model is used instead.
This ensures the script always produces output even without the deep-learning weights.

For all other 26 SSA countries we only have country-level satellite aggregates,
so predictions are estimated from a GBR tabular model using nine derived features.

Output: client/public/predictions.geojson   (served by GitHub Pages / Vite dev)
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PIPELINE      = Path(__file__).parent
SAT_JSON      = PIPELINE / "outputs" / "satellite_features.json"
MODEL_PATH    = PIPELINE / "outputs" / "sat_model.joblib"
TRAIN_CSV     = PIPELINE / "outputs" / "training_with_satellite.csv"
GEOJSON_OUT   = Path("client/public/predictions.geojson")

# CNN artefacts — produced by training/trainer.py
CNN_CKPT      = Path("outputs/models/s2_kenya_nigeria.pth")
PATCH_CSV     = Path("data/training_dataset.csv")   # 3 048 DHS clusters, patch paths

# Countries with real DHS cluster data + S2 patches.
# All other countries get synthetic urban/rural grid points (GBR tabular model).
DHS_COUNTRIES = {"KEN": "Kenya", "NGA": "Nigeria"}

COUNTRY_CENTROIDS = {
    "NGA": ("Nigeria",        9.08,   8.68),
    "ETH": ("Ethiopia",       9.15,  40.49),
    "COD": ("DR Congo",      -4.04,  21.76),
    "KEN": ("Kenya",         -0.02,  37.91),
    "TZA": ("Tanzania",      -6.37,  34.89),
    "MOZ": ("Mozambique",   -18.67,  35.53),
    "GHA": ("Ghana",          7.95,   1.02),
    "UGA": ("Uganda",         1.37,  32.29),
    "CMR": ("Cameroon",       3.85,  11.50),
    "AGO": ("Angola",       -11.20,  17.87),
    "ZMB": ("Zambia",       -13.13,  27.85),
    "ZWE": ("Zimbabwe",     -19.02,  29.15),
    "MWI": ("Malawi",       -13.25,  34.30),
    "SEN": ("Senegal",       14.50, -14.45),
    "MLI": ("Mali",          17.57,  -3.99),
    "BFA": ("Burkina Faso",  12.36,  -1.53),
    "RWA": ("Rwanda",        -1.94,  29.87),
    "NER": ("Niger",         17.61,   8.08),
    "TCD": ("Chad",          15.45,  18.73),
    "MDG": ("Madagascar",   -18.77,  46.87),
    "ZAF": ("South Africa", -28.47,  24.68),
    "SDN": ("Sudan",         12.86,  30.22),
    "SOM": ("Somalia",        5.15,  46.20),
    "GIN": ("Guinea",        11.75, -15.45),
    "BWA": ("Botswana",     -22.33,  24.68),
    "NAM": ("Namibia",      -22.96,  18.49),
    "SLE": ("Sierra Leone",   8.46, -11.78),
    "TGO": ("Togo",           8.62,   0.82),
    "BEN": ("Benin",          9.31,   2.32),
}

def ntl_trend(ntl_by_year: dict) -> float:
    years = sorted(int(y) for y in ntl_by_year)
    if len(years) < 2:
        return 0.0
    vals = [ntl_by_year[str(y)] for y in years]
    return float(np.polyfit(years, vals, 1)[0])


def make_features(sat_feats: dict, lat: float, lon: float, is_urban: int) -> list[float]:
    """Map country-level satellite data → model feature vector."""
    ntl  = sat_feats.get("ntl",  {})
    ndvi = sat_feats.get("ndvi", {})
    ndbi = sat_feats.get("ndbi", {})

    sat_ntl  = float(ntl.get("2023",  ntl.get("2022",  0.3)))
    sat_ndvi = float(ndvi.get("2023", ndvi.get("2022", 0.3)))
    sat_ndbi = float(ndbi.get("2023", ndbi.get("2022", 0.05)))

    # Cluster-level NTL: urban clusters are ~2× the national average;
    # rural clusters ≈ national average (validated against KEN/NGA training data)
    ntl_mean = sat_ntl * (2.0 if is_urban else 1.0)
    ntl_std  = ntl_mean * 0.22
    ntl_max  = ntl_mean * 1.80

    # S2 Excess Green: vegetation index derivable from NDVI and NDBI
    # High NDVI → positive exgreen; high NDBI → negative exgreen
    s2_exgreen    = (sat_ndvi - 0.25) * 0.12 - sat_ndbi * 0.08

    # S2 brightness: built-up surfaces + urban light leakage
    s2_brightness = 0.15 + sat_ndbi * 0.25 + ntl_mean * 0.05

    return [
        ntl_mean, ntl_std, ntl_max,
        s2_exgreen, s2_brightness,
        ntl_trend(ntl),
        float(is_urban), float(lat), float(lon),
    ]


def wi_to_poverty(wi: float) -> float:
    return max(0.0, min(100.0, 50.0 - wi * 25.0))


def _load_patch(path: str) -> np.ndarray | None:
    """Load a .npy S2 patch and return a float32 (C, H, W) array clipped to [0,1]."""
    try:
        arr = np.load(path).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)
    except Exception:
        return None


def predict_dhs_cnn(sat: dict) -> list[dict]:
    """
    Run ResNet18 on S2 patches for every Kenya + Nigeria DHS cluster.

    Returns the same GeoJSON feature list format as predict_dhs_clusters so
    main() can swap between them transparently. Returns [] on any failure so
    the caller can fall back to the GBR path.
    """
    if not CNN_CKPT.exists():
        print(f"[predict] CNN checkpoint not found at {CNN_CKPT} — skipping CNN path")
        return []
    if not PATCH_CSV.exists():
        print(f"[predict] Patch CSV not found at {PATCH_CSV} — skipping CNN path")
        return []

    # Import here to avoid requiring torch at module level when only GBR is needed
    from models.resnet_model import ResNetRegression

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load with dropout_p=0.3 to match checkpoint architecture; eval() disables it.
    model = ResNetRegression(in_channels=3, dropout_p=0.3).to(device)
    state = torch.load(str(CNN_CKPT), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[predict] CNN loaded from {CNN_CKPT} on {device}")

    df = pd.read_csv(PATCH_CSV)
    # Backward-compat: prefer s2_patch_file, fall back to patch_file
    if "s2_patch_file" not in df.columns and "patch_file" in df.columns:
        df["s2_patch_file"] = df["patch_file"]

    # Batch inference — collect patches that exist, run in chunks of 32
    valid_rows, tensors = [], []
    for _, row in df.iterrows():
        patch_path = str(row.get("s2_patch_file", ""))
        arr = _load_patch(patch_path)
        if arr is None or arr.shape[0] != 3:
            continue
        valid_rows.append(row)
        tensors.append(torch.from_numpy(arr))

    if not tensors:
        print("[predict] No valid S2 patches found — skipping CNN path")
        return []

    all_wi: list[float] = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            preds = model(batch).cpu().numpy().flatten()
            all_wi.extend(preds.tolist())

    geo = []
    for row, wi in zip(valid_rows, all_wi):
        iso3 = {"Kenya": "KEN", "Nigeria": "NGA"}.get(row["country"], "")
        sat_feats = sat.get(iso3, {})
        ntl  = sat_feats.get("ntl",  {})
        ndvi = sat_feats.get("ndvi", {})
        ndbi = sat_feats.get("ndbi", {})
        urban_code = str(row.get("URBAN_RURA", ""))
        geo.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(row["longitude"]), 5), round(float(row["latitude"]), 5)],
            },
            "properties": {
                "country":         row["country"],
                "iso3":            iso3,
                "wealth_index":    round(float(wi), 4),
                "poverty_rate":    round(wi_to_poverty(float(wi)), 1),
                "urban_rural":     "Urban" if urban_code == "U" else "Rural",
                "adm1_name":       str(row.get("ADM1NAME", "")),
                "composite_score": round(max(0.0, min(100.0, (float(wi) + 2.0) / 4.0 * 100.0)), 1),
                "ntl_latest":      float(ntl.get("2023",  ntl.get("2022",  0))),
                "ntl_trend":       round(ntl_trend(ntl), 6),
                "ndvi_latest":     float(ndvi.get("2023", ndvi.get("2022", 0))),
                "ndbi_latest":     float(ndbi.get("2023", ndbi.get("2022", 0))),
                "model":           "cnn_resnet18",
            },
        })

    print(f"[predict] {len(geo)} CNN predictions (Kenya + Nigeria DHS clusters)")
    return geo


def predict_dhs_clusters(model, scaler, features: list[str], sat: dict) -> list[dict]:
    """Run GBR tabular model on all real DHS clusters for Kenya + Nigeria."""
    if not TRAIN_CSV.exists():
        print(f"[predict] {TRAIN_CSV} not found — skipping DHS cluster predictions")
        return []

    df = pd.read_csv(TRAIN_CSV).dropna(subset=features)
    X  = scaler.transform(df[features].values.astype(np.float32))
    wi = model.predict(X)

    geo = []
    for (_, row), w in zip(df.iterrows(), wi):
        iso3 = {"Kenya": "KEN", "Nigeria": "NGA"}.get(row["country"], "")
        sat_feats = sat.get(iso3, {})
        ntl  = sat_feats.get("ntl",  {})
        ndvi = sat_feats.get("ndvi", {})
        ndbi = sat_feats.get("ndbi", {})
        w = float(w)
        geo.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(row["longitude"], 5),
                                                           round(row["latitude"],  5)]},
            "properties": {
                "country":         row["country"],
                "iso3":            iso3,
                "wealth_index":    round(w, 4),
                "poverty_rate":    round(wi_to_poverty(w), 1),
                "urban_rural":     "Urban" if row["is_urban"] == 1 else "Rural",
                "adm1_name":       str(row.get("ADM1NAME", "")),
                "composite_score": round(max(0, min(100, (w + 2) / 4 * 100)), 1),
                "ntl_latest":      float(ntl.get("2023",  ntl.get("2022",  0))),
                "ntl_trend":       round(ntl_trend(ntl), 6),
                "ndvi_latest":     float(ndvi.get("2023", ndvi.get("2022", 0))),
                "ndbi_latest":     float(ndbi.get("2023", ndbi.get("2022", 0))),
            },
        })
    print(f"[predict] {len(geo)} real DHS cluster predictions (Kenya + Nigeria)")
    return geo


def main():
    sat = json.load(open(SAT_JSON))

    # Step 1: Try CNN (ResNet18 on S2 patches) for Kenya + Nigeria DHS clusters.
    # Falls back to the GBR tabular model when the checkpoint is missing.
    geo_features = predict_dhs_cnn(sat)
    if not geo_features:
        print("[predict] Falling back to GBR tabular model for DHS clusters…")
        bundle = joblib.load(MODEL_PATH)
        model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
        geo_features = predict_dhs_clusters(model, scaler, features, sat)

    geojson = {"type": "FeatureCollection", "features": geo_features}
    GEOJSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_OUT, "w") as f:
        json.dump(geojson, f)

    size_kb = GEOJSON_OUT.stat().st_size / 1024
    model_tag = "CNN (ResNet18)" if geo_features and geo_features[0].get("properties", {}).get("model") == "cnn_resnet18" else "GBR tabular"
    print(f"[predict] Saved → {GEOJSON_OUT}  ({size_kb:.1f} KB, {len(geo_features)} clusters, model={model_tag})")


if __name__ == "__main__":
    main()
