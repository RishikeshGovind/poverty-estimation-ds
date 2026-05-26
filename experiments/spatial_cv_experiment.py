"""
Spatial cross-validation experiment.

Trains and evaluates the multi-task model using leave-one-country-out splits,
measuring how well the model generalises to unseen geographic regions.

Usage:
    python -m experiments.spatial_cv_experiment
    python -m experiments.spatial_cv_experiment --epochs 5   # quick test
    python -m experiments.spatial_cv_experiment --sensors s2 s1
"""

import argparse
import os
import pandas as pd

from training.spatial_cv import assign_countries, leave_one_country_out
from training.multitask_trainer import run_multitask_training
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def main(sensors=None, epochs=None):
    cfg = load_config()
    csv_path = cfg["data"]["training_csv"]
    out_path = cfg["spatial_cv"]["output"]
    ckpt_dir = os.path.join(cfg["training"]["model_dir"], "spatial_cv")
    sensors  = sensors or ["s2"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(csv_path):
        logger.error("Training CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Ensure SDG labels are present
    for col in ["sdg1_wealth", "sdg7_ntl", "sdg11_buildup"]:
        if col not in df.columns:
            logger.warning("Column %s missing — run compute_sdg_labels.py first. Adding NaN.", col)
            df[col] = float("nan")

    df = assign_countries(df)
    df.to_csv(csv_path, index=False)   # persist country column

    results = []
    for country, train_idx, val_idx in leave_one_country_out(df):
        ckpt = os.path.join(ckpt_dir, f"held_out_{country.replace(' ', '_')}.pth")
        try:
            metrics, _ = run_multitask_training(
                sensors=sensors,
                run_name=f"hold-out={country}",
                epochs=epochs,
                csv_path=csv_path,
                checkpoint_path=ckpt,
                train_indices=train_idx,
                val_indices=val_idx,
            )
            metrics["held_out_country"] = country
            metrics["n_train"] = len(train_idx)
            metrics["n_test"]  = len(val_idx)
            results.append(metrics)
        except Exception as e:
            logger.error("Fold %s failed: %s", country, e)

    if not results:
        logger.error("No folds completed — need data from multiple countries.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    logger.info("Spatial CV results saved to %s", out_path)

    # Summary: mean R² across folds for primary task
    if "sdg1_wealth_r2" in results_df.columns:
        mean_r2 = results_df["sdg1_wealth_r2"].mean()
        std_r2  = results_df["sdg1_wealth_r2"].std()
        logger.info("Spatial CV  SDG1 wealth R²: %.4f ± %.4f", mean_r2, std_r2)

    logger.info("\n%s", results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="e.g. --sensors s2 s1 viirs")
    args = parser.parse_args()
    main(sensors=args.sensors, epochs=args.epochs)
