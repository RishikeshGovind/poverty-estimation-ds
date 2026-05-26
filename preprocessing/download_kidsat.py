"""
Phase 1 — KidSat dataset downloader.

KidSat (Alan Turing Institute, 2024) provides 33,608 Sentinel-2 + Landsat
images (10×10 km) for 19 African countries paired with DHS child poverty labels.
This is the fastest path to real training data while DHS approval is pending.

Paper : https://arxiv.org/abs/2407.05986
GitHub: https://github.com/MLGlobalHealth/KidSat

The dataset is hosted on Hugging Face:
  MLGlobalHealth/KidSat  — images + label CSV

Run:
    python -m preprocessing.download_kidsat
"""

import os
import shutil
import pandas as pd

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def download_kidsat(output_dir: str = None, max_samples: int = None):
    """
    Download KidSat satellite images and labels from Hugging Face.

    Args:
        output_dir:   where to save images (default: data/kidsat/)
        max_samples:  limit download size for testing (None = all 33k)

    Returns:
        Path to the generated training CSV.
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    cfg = load_config()
    output_dir = output_dir or os.path.join("data", "kidsat")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Downloading KidSat label CSV from Hugging Face…")

    # Download just the CSV first to inspect without pulling all images
    label_csv = hf_hub_download(
        repo_id="MLGlobalHealth/KidSat",
        filename="labels/dhs_final_labels.csv",
        repo_type="dataset",
        local_dir=output_dir,
    )
    logger.info("Labels downloaded to %s", label_csv)

    df = pd.read_csv(label_csv)
    logger.info("KidSat: %d total records, columns: %s", len(df), list(df.columns))

    # Optional: limit for quick testing
    if max_samples:
        df = df.head(max_samples)
        logger.info("Limiting to %d samples for testing", max_samples)

    # Download images in batches
    # KidSat images are stored as: images/<survey_id>/<cluster_id>.npy
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Build the training CSV in our pipeline's format
    rows = []
    downloaded = 0
    skipped = 0

    for _, row in df.iterrows():
        # KidSat column names (adjust if the CSV schema changes)
        survey_id  = str(row.get("survey_id",  row.get("SurveyId",  "")))
        cluster_id = str(row.get("cluster_id", row.get("DHSCLUST",  "")))
        lat  = float(row.get("lat",  row.get("LATNUM",  0)))
        lon  = float(row.get("lon",  row.get("LONGNUM", 0)))
        # Wealth score — KidSat uses MICS/DHS child poverty index
        label = float(row.get("wealth_index", row.get("wi", row.get("iwi", 0))))

        # Build expected HF path
        hf_path = f"images/{survey_id}/{cluster_id}.npy"
        local_path = os.path.join(images_dir, survey_id, f"{cluster_id}.npy")

        if not os.path.exists(local_path):
            try:
                hf_hub_download(
                    repo_id="MLGlobalHealth/KidSat",
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir=output_dir,
                )
                downloaded += 1
            except Exception as e:
                logger.debug("Could not download %s: %s", hf_path, e)
                skipped += 1
                local_path = ""

        rows.append({
            "latitude":       lat,
            "longitude":      lon,
            "label":          label,
            "country":        row.get("country", row.get("CountryName", "")),
            "s2_patch_file":  local_path if os.path.exists(local_path) else "",
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "kidsat_training.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(
        "KidSat CSV written: %d rows | downloaded=%d  skipped=%d → %s",
        len(out_df), downloaded, skipped, out_csv,
    )
    return out_csv


def merge_with_existing(kidsat_csv: str, existing_csv: str, output_csv: str = None):
    """
    Merge KidSat labels with an existing training CSV so both data sources
    train together.
    """
    df_new = pd.read_csv(kidsat_csv)
    df_old = pd.read_csv(existing_csv)

    # Align columns — add missing ones as empty strings / NaN
    for col in df_old.columns:
        if col not in df_new.columns:
            df_new[col] = ""
    for col in df_new.columns:
        if col not in df_old.columns:
            df_old[col] = ""

    merged = pd.concat([df_old, df_new], ignore_index=True)
    output_csv = output_csv or existing_csv
    merged.to_csv(output_csv, index=False)
    logger.info("Merged %d (existing) + %d (KidSat) = %d rows → %s",
                len(df_old), len(df_new), len(merged), output_csv)
    return output_csv


if __name__ == "__main__":
    csv = download_kidsat(max_samples=100)   # remove limit for full download
    print(f"KidSat training CSV: {csv}")
