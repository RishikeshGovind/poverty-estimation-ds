"""
Build data/training_dataset.csv by joining S2 + VIIRS patch indices for all
configured countries. Missing sensor columns are left blank (dataset fills with zeros).

Usage:
  python -m preprocessing.build_training_csv
  python -m preprocessing.build_training_csv --countries Kenya Nigeria
"""

import argparse
import os
import pandas as pd

SENSORS = {
    "s2":    ("data/patches/s2_{country}/index.csv",    "s2_patch_file",    "patch_path"),
    "viirs": ("data/patches/viirs_{country}/index.csv", "viirs_patch_file", "patch_path"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=["Kenya", "Nigeria"])
    parser.add_argument("--out", default="data/training_dataset.csv")
    args = parser.parse_args()

    frames = []
    for country in args.countries:
        s2_csv = f"data/patches/s2_{country.lower()}/index.csv"
        if not os.path.exists(s2_csv):
            print(f"Skipping {country} — S2 index not found: {s2_csv}")
            continue

        df = pd.read_csv(s2_csv).rename(columns={"patch_path": "s2_patch_file"})

        # Join VIIRS if available
        viirs_csv = f"data/patches/viirs_{country.lower()}/index.csv"
        if os.path.exists(viirs_csv):
            viirs = pd.read_csv(viirs_csv)[["cluster_id", "patch_path"]].rename(
                columns={"patch_path": "viirs_patch_file"}
            )
            df = df.merge(viirs, on="cluster_id", how="left")
            print(f"{country}: {len(df)} S2 patches, "
                  f"{df['viirs_patch_file'].notna().sum()} VIIRS patches")
        else:
            df["viirs_patch_file"] = ""
            print(f"{country}: {len(df)} S2 patches, no VIIRS yet")

        df["s1_patch_file"]   = ""
        df["patch_file"]      = df["s2_patch_file"]   # backward compat
        df["label"]           = df["wealth_index"]

        frames.append(df)

    if not frames:
        print("No data found — run download_s2_patches.py first.")
        return

    combined = pd.concat(frames, ignore_index=True)

    base_cols = ["longitude","latitude","wealth_index","country","label",
                 "s2_patch_file","s1_patch_file","viirs_patch_file","patch_file"]
    extra_cols = [c for c in ["ADM1NAME","DHSREGNA","URBAN_RURA"] if c in combined.columns]
    combined[base_cols + extra_cols].to_csv(args.out, index=False)

    print(f"\nWrote {len(combined)} rows to {args.out}")
    print(combined.groupby("country")["wealth_index"].describe().round(3))


if __name__ == "__main__":
    main()
