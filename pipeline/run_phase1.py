"""
Run all Phase 1 extractions in sequence and merge into satellite_features.json.

Usage:
    python pipeline/run_phase1.py                     # full 2014–2024 run
    python pipeline/run_phase1.py --years 2022 2023   # quick test with 2 years

Output: pipeline/outputs/satellite_features.json
Schema:
  {
    "NGA": {
      "ntl":  {"2014": 1.23, "2015": 1.41, ...},   # VIIRS mean radiance nW/cm²/sr
      "ndvi": {"2019": 0.42, "2020": 0.40, ...},   # Sentinel-2 NDVI (veg health)
      "ndbi": {"2019": -0.08, ...},                 # Sentinel-2 NDBI (urban index)
      "ndvi_landsat": {"2014": 0.38, ...}           # Landsat NDVI (longer history)
    },
    ...
  }
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import VIIRS_OUT, SENTINEL_OUT, LANDSAT_OUT, MERGED_OUT, VIIRS_YEARS


def run_script(script: str, years: list[int]):
    """Run a pipeline script as a subprocess so each gets a clean GEE session."""
    cmd = [sys.executable, script, "--years"] + [str(y) for y in years]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"[warn] {script} exited with code {result.returncode} — continuing")


def merge_outputs():
    """Combine the three JSON outputs into one satellite_features.json."""
    merged: dict = {}

    def load(path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        print(f"[merge] Missing {path.name} — skipping")
        return {}

    viirs    = load(VIIRS_OUT)
    sentinel = load(SENTINEL_OUT)
    landsat  = load(LANDSAT_OUT)

    all_iso3 = set(viirs) | set(sentinel) | set(landsat)
    for iso3 in sorted(all_iso3):
        entry: dict = {}
        if iso3 in viirs:
            entry["ntl"] = viirs[iso3]
        if iso3 in sentinel:
            entry["ndvi"] = sentinel[iso3].get("ndvi", {})
            entry["ndbi"] = sentinel[iso3].get("ndbi", {})
        if iso3 in landsat:
            entry["ndvi_landsat"] = landsat[iso3]
        merged[iso3] = entry

    with open(MERGED_OUT, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n[merge] satellite_features.json → {MERGED_OUT}")
    print(f"[merge] {len(merged)} countries, fields per country: ntl / ndvi / ndbi / ndvi_landsat")
    return merged


def main(years: list[int]):
    base = Path(__file__).parent

    run_script(str(base / "extract_viirs.py"),     years)
    run_script(str(base / "extract_sentinel2.py"), [y for y in years if y >= 2019])
    run_script(str(base / "extract_landsat.py"),   years)

    merged = merge_outputs()

    # Quick sanity check
    if merged:
        iso3   = next(iter(merged))
        fields = list(merged[iso3].keys())
        print(f"\n[ok] Phase 1 complete. Sample ({iso3}): fields={fields}")
        for field, vals in merged[iso3].items():
            latest_year = max(vals.keys()) if vals else "—"
            latest_val  = vals.get(latest_year, "—")
            print(f"     {field}: {latest_year} → {latest_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=VIIRS_YEARS,
                        help="Years to extract (default 2014–2024). Use 2 years for a quick test.")
    args = parser.parse_args()
    main(args.years)
