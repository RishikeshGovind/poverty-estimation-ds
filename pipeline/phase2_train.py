"""
Phase 2 Task 6 — Train on cluster-level satellite + DHS features.

Features (all vary per cluster, not per country):
  ntl_mean      — mean VIIRS radiance in 64×64 patch  (electrification)
  ntl_std       — spatial std of NTL (urban structure)
  ntl_max       — peak NTL in patch (commercial/industrial hotspot)
  s2_exgreen    — Excess Green = 2G−R−B (vegetation health proxy w/o NIR)
  s2_brightness — mean visible reflectance (surface albedo / urban proxy)
  ntl_trend     — country NTL slope 2019→2023 (economic trajectory)
  is_urban      — DHS urban/rural flag
  latitude      — cluster latitude
  longitude     — cluster longitude

Label: wealth_index (DHS continuous, ~−3 to 3)
Model: GradientBoostingRegressor
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PIPELINE    = Path(__file__).parent
TRAIN_CSV   = PIPELINE / "outputs" / "training_with_satellite.csv"
MODEL_OUT   = PIPELINE / "outputs" / "sat_model.joblib"
METRICS_OUT = PIPELINE / "outputs" / "model_metrics.json"

FEATURES = [
    "ntl_mean", "ntl_std", "ntl_max",
    "s2_exgreen", "s2_brightness",
    "ntl_trend",
    "is_urban", "latitude", "longitude",
]


def main():
    print("[train] Loading enriched training data…")
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=FEATURES + ["wealth_index"])
    print(f"[train] {len(df)} rows after dropping NaNs")

    # 80/20 stratified by country
    train_df = df.groupby("country", group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42)
    )
    val_df = df.drop(train_df.index)

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df["wealth_index"].values.astype(np.float32)
    X_val   = val_df[FEATURES].values.astype(np.float32)
    y_val   = val_df["wealth_index"].values.astype(np.float32)

    print(f"[train] Train: {len(X_train)}  Val: {len(X_val)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    print("[train] Fitting GradientBoostingRegressor…")
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    r2   = r2_score(y_val, preds)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)

    print(f"\n[train] R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    print("[train] Feature importances:")
    importances = dict(zip(FEATURES, model.feature_importances_.round(4)))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<18} {imp:.4f}  {bar}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, MODEL_OUT)
    print(f"\n[train] Model → {MODEL_OUT}")

    metrics = {
        "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
        "n_train": len(X_train), "n_val": len(X_val),
        "feature_importances": importances,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics → {METRICS_OUT}")
    return metrics


if __name__ == "__main__":
    main()
