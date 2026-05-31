"""
Phase 2 Task 7 — Generate predictions.geojson for all 30 SSA countries.

For each country in satellite_features.json:
  1. Create a grid of sample points (urban near centroid, rural spread out)
  2. Predict wealth_index using the trained GBM model
  3. Convert to poverty_rate = max(0, min(100, 50 - wi * 25))
     (matches formula in client/src/hooks/useModelPredictions.ts)

Output: client/public/predictions.geojson  (read by the dashboard)
"""

import json
import math
import joblib
import numpy as np
from pathlib import Path

PIPELINE    = Path(__file__).parent
SAT_JSON    = PIPELINE / "outputs" / "satellite_features.json"
MODEL_PATH  = PIPELINE / "outputs" / "sat_model.joblib"
GEOJSON_OUT = Path("client/public/predictions.geojson")

# Country centroids (lat, lon) — from config.py SSA_COUNTRIES
COUNTRY_CENTROIDS = {
    "NGA": ("Nigeria",       9.08,   8.68),
    "ETH": ("Ethiopia",      9.15,  40.49),
    "COD": ("DR Congo",     -4.04,  21.76),
    "KEN": ("Kenya",        -0.02,  37.91),
    "TZA": ("Tanzania",     -6.37,  34.89),
    "MOZ": ("Mozambique",  -18.67,  35.53),
    "GHA": ("Ghana",         7.95,   1.02),
    "UGA": ("Uganda",        1.37,  32.29),
    "CMR": ("Cameroon",      3.85,  11.50),
    "AGO": ("Angola",      -11.20,  17.87),
    "ZMB": ("Zambia",      -13.13,  27.85),
    "ZWE": ("Zimbabwe",    -19.02,  29.15),
    "MWI": ("Malawi",      -13.25,  34.30),
    "SEN": ("Senegal",      14.50, -14.45),
    "MLI": ("Mali",         17.57,  -3.99),
    "BFA": ("Burkina Faso", 12.36,  -1.53),
    "RWA": ("Rwanda",       -1.94,  29.87),
    "NER": ("Niger",        17.61,   8.08),
    "TCD": ("Chad",         15.45,  18.73),
    "MDG": ("Madagascar",  -18.77,  46.87),
    "ZAF": ("South Africa",-28.47,  24.68),
    "SDN": ("Sudan",        12.86,  30.22),
    "SOM": ("Somalia",       5.15,  46.20),
    "GIN": ("Guinea",       11.75, -15.45),
    "BWA": ("Botswana",    -22.33,  24.68),
    "NAM": ("Namibia",     -22.96,  18.49),
    "SLE": ("Sierra Leone",  8.46, -11.78),
    "TGO": ("Togo",          8.62,   0.82),
    "BEN": ("Benin",         9.31,   2.32),
    "HTI": ("Haiti",        18.97, -72.29),
}

N_URBAN  = 20   # sample points per country (urban)
N_RURAL  = 40   # sample points per country (rural)
RNG      = np.random.default_rng(42)


def ntl_trend(ntl_by_year: dict) -> float:
    years = sorted(int(y) for y in ntl_by_year)
    if len(years) < 2:
        return 0.0
    vals = [ntl_by_year[str(y)] for y in years]
    return float(np.polyfit(years, vals, 1)[0])


def make_features(iso3: str, feats: dict, lat: float, lon: float, is_urban: int) -> list:
    ntl  = feats.get("ntl", {})
    ndvi = feats.get("ndvi", {})
    ndbi = feats.get("ndbi", {})
    return [
        float(ntl.get("2023", ntl.get("2022", 0.0))),
        ntl_trend(ntl),
        float(ndvi.get("2023", ndvi.get("2022", 0.0))),
        float(ndbi.get("2023", ndbi.get("2022", 0.0))),
        float(is_urban),
        float(lat),
        float(lon),
    ]


def wi_to_poverty_rate(wi: float) -> float:
    return max(0.0, min(100.0, 50.0 - wi * 25.0))


def main():
    print("[predict] Loading model and satellite features…")
    bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    sat = json.load(open(SAT_JSON))

    features_list = []
    meta_list     = []

    for iso3, (country_name, clat, clon) in COUNTRY_CENTROIDS.items():
        feats = sat.get(iso3, {})

        # Urban sample: tight cluster near centroid
        for _ in range(N_URBAN):
            lat = clat + RNG.normal(0, 0.3)
            lon = clon + RNG.normal(0, 0.3)
            features_list.append(make_features(iso3, feats, lat, lon, 1))
            meta_list.append((country_name, iso3, lat, lon, "U"))

        # Rural sample: spread across the country
        for _ in range(N_RURAL):
            lat = clat + RNG.uniform(-4, 4)
            lon = clon + RNG.uniform(-4, 4)
            features_list.append(make_features(iso3, feats, lat, lon, 0))
            meta_list.append((country_name, iso3, lat, lon, "R"))

    X = scaler.transform(np.array(features_list, dtype=np.float32))
    preds = model.predict(X)

    print(f"[predict] Generated {len(preds)} cluster predictions across {len(COUNTRY_CENTROIDS)} countries")

    features_geojson = []
    for (country, iso3, lat, lon, ur), wi in zip(meta_list, preds):
        poverty_rate = wi_to_poverty_rate(float(wi))
        features_geojson.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
            "properties": {
                "country":         country,
                "wealth_index":    round(float(wi), 4),
                "composite_score": round((float(wi) + 2) / 4, 4),
                "poverty_rate":    round(poverty_rate, 1),
                "urban_rural":     ur,
                "adm1_name":       "",
                "iso3":            iso3,
            }
        })

    geojson = {"type": "FeatureCollection", "features": features_geojson}

    GEOJSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_OUT, "w") as f:
        json.dump(geojson, f)

    size_kb = GEOJSON_OUT.stat().st_size / 1024
    print(f"[predict] Saved → {GEOJSON_OUT}  ({size_kb:.1f} KB)")

    # Print summary stats per country
    from collections import defaultdict
    by_country: dict[str, list] = defaultdict(list)
    for (_, iso3, _, _, _), wi in zip(meta_list, preds):
        by_country[iso3].append(wi_to_poverty_rate(float(wi)))

    print("\n[predict] Country poverty rate estimates (model):")
    for iso3, rates in sorted(by_country.items()):
        name = COUNTRY_CENTROIDS[iso3][0]
        print(f"  {iso3} {name:<20} avg={sum(rates)/len(rates):.1f}%  "
              f"urban={sum(r for r, (_, _, _, _, ur) in zip(rates, [m for m in meta_list if m[1]==iso3]) if ur=='U')/N_URBAN:.1f}%")


if __name__ == "__main__":
    main()
