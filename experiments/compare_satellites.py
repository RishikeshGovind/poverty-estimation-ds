"""
Satellite comparison experiment.

Trains one ResNet model per sensor combination (same architecture, same
hyperparameters, same DHS labels) and records R², RMSE, MAE for each.

Combinations:
  S2-only        3 ch  (optical RGB)
  S1-only        2 ch  (SAR VV+VH)
  VIIRS-only     1 ch  (nighttime lights)
  S2+S1          5 ch
  S2+VIIRS       4 ch
  S2+S1+VIIRS    6 ch  (full fusion)

Output: outputs/experiments/satellite_comparison.csv  (and per-run checkpoints)

Usage:
    python -m experiments.compare_satellites
    python -m experiments.compare_satellites --epochs 10   # quick test
"""

import argparse
import os
import pandas as pd

from training.trainer import run_training
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

COMBOS = [
    (["s2"],              "S2-only"),
    (["s1"],              "S1-only"),
    (["viirs"],           "VIIRS-only"),
    (["s2", "s1"],        "S2+S1"),
    (["s2", "viirs"],     "S2+VIIRS"),
    (["s2", "s1", "viirs"], "S2+S1+VIIRS"),
]


def main(epochs=None, skip_missing=True):
    cfg = load_config()
    csv_path   = cfg["data"]["training_csv"]
    ckpt_dir   = os.path.join(cfg["training"]["model_dir"], "experiments")
    out_dir    = "outputs/experiments"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        logger.error("Training CSV not found: %s — run preprocessing first.", csv_path)
        return

    # Detect which sensors have patch data in the CSV
    import pandas as _pd
    df = _pd.read_csv(csv_path)
    available = {"s2"}   # S2 always required
    if "s1_patch_file" in df.columns and df["s1_patch_file"].notna().any() and (df["s1_patch_file"] != "").any():
        available.add("s1")
    if "viirs_patch_file" in df.columns and df["viirs_patch_file"].notna().any() and (df["viirs_patch_file"] != "").any():
        available.add("viirs")

    logger.info("Available sensors in dataset: %s", sorted(available))

    results = []
    for sensors, name in COMBOS:
        missing = [s for s in sensors if s not in available]
        if missing:
            if skip_missing:
                logger.warning("Skipping %s — sensor(s) %s not in dataset.", name, missing)
                continue
            # Fill missing sensors with zeros — still runs, just less informative
            logger.warning("Running %s with missing sensor(s) %s filled with zeros.", name, missing)

        try:
            metrics = run_training(
                sensors=sensors,
                run_name=name,
                epochs=epochs,
                csv_path=csv_path,
                checkpoint_dir=ckpt_dir,
            )
            results.append(metrics)
        except Exception as e:
            logger.error("Run %s failed: %s", name, e)

    if not results:
        logger.error("No runs completed.")
        return

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    out_csv = os.path.join(out_dir, "satellite_comparison.csv")
    results_df.to_csv(out_csv, index=False)

    logger.info("\n%s", results_df.to_string(index=False))
    logger.info("Results saved to %s", out_csv)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from config.yaml")
    parser.add_argument("--include-missing", action="store_true",
                        help="Run combos even if sensor patches are absent (filled with zeros)")
    args = parser.parse_args()
    main(epochs=args.epochs, skip_missing=not args.include_missing)
