"""
Phase 2 Task 6 — Train gradient boosting model on satellite + DHS features.

Features (all tabular, no CNN needed):
  ntl_2023     — country mean nighttime lights radiance (electrification proxy)
  ntl_trend    — NTL slope 2019→2023 (economic trajectory)
  ndvi_2023    — country mean vegetation index (food security proxy)
  ndbi_2023    — country mean built-up index (urbanisation proxy)
  is_urban     — 1=urban cluster, 0=rural
  latitude     — cluster latitude
  longitude    — cluster longitude

Label: wealth_index (DHS, continuous ~-3 to 3)

Model: GradientBoostingRegressor (sklearn, fast, no GPU needed)

Outputs:
  pipeline/outputs/sat_model.joblib   — trained model + scaler
  pipeline/outputs/model_metrics.json — R², RMSE, MAE on val set
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

PIPELINE   = Path(__file__).parent
TRAIN_CSV  = PIPELINE / "outputs" / "training_with_satellite.csv"
MODEL_OUT  = PIPELINE / "outputs" / "sat_model.joblib"
METRICS_OUT= PIPELINE / "outputs" / "model_metrics.json"

FEATURES = ["ntl_2023", "ntl_trend", "ndvi_2023", "ndbi_2023", "is_urban", "latitude", "longitude"]


def main():
    print("[train] Loading enriched training data…")
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=FEATURES + ["wealth_index"])

    # 80 / 20 stratified by country so both countries appear in val set
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
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    r2   = r2_score(y_val, preds)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)

    print(f"[train] R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # Feature importances
    importances = dict(zip(FEATURES, model.feature_importances_.round(4)))
    print("[train] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"         {feat:<15} {imp:.4f}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, MODEL_OUT)
    print(f"[train] Model saved → {MODEL_OUT}")

    metrics = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
               "n_train": len(X_train), "n_val": len(X_val),
               "feature_importances": importances}
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved → {METRICS_OUT}")
    return metrics


if __name__ == "__main__":
    main()
