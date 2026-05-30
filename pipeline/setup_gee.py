"""
Task 1 — Google Earth Engine setup and authentication.

Run this ONCE interactively:
    python pipeline/setup_gee.py

Steps it performs:
  1. Checks earthengine-api is installed
  2. Opens browser for OAuth (or uses service-account key if GEE_KEY_FILE is set)
  3. Saves credentials to ~/.config/earthengine/credentials
  4. Runs a test query to confirm access

For CI / GitHub Actions use a service account instead:
  - Create a service account in https://console.cloud.google.com
  - Grant it "Earth Engine Resource Viewer" role
  - Download the JSON key → set GEE_KEY_FILE=/path/to/key.json
  - Set GEE_PROJECT=your-cloud-project-id
"""

import os
import sys


def check_install():
    try:
        import ee
        print(f"[ok] earthengine-api installed")
        return ee
    except ImportError:
        print("[error] earthengine-api not installed.")
        print("        Run: pip install earthengine-api")
        sys.exit(1)


def authenticate(ee):
    key_file = os.environ.get("GEE_KEY_FILE", "")
    project  = os.environ.get("GEE_PROJECT", "")

    if key_file and os.path.exists(key_file):
        # Service-account authentication (headless / CI)
        credentials = ee.ServiceAccountCredentials(
            email=None,   # read from key file
            key_file=key_file,
        )
        ee.Initialize(credentials=credentials, project=project or None)
        print(f"[ok] Authenticated via service account: {key_file}")
    else:
        # Interactive browser-based OAuth
        print("[info] Opening browser for Google Earth Engine authentication…")
        print("       If no browser opens, copy the URL printed below and paste it manually.")
        ee.Authenticate()
        ee.Initialize(project=project or None)
        print("[ok] Authenticated via OAuth")


def test_connection(ee):
    print("[info] Testing GEE connection…")
    # Fetch one VIIRS tile — should return instantly
    img = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG") \
             .filterDate("2023-01-01", "2023-02-01") \
             .select("avg_rad") \
             .first()
    info = img.getInfo()
    if info:
        print(f"[ok] GEE connection works — got image: {info.get('id', 'unknown')}")
    else:
        print("[warn] Connection test returned empty result — check your project settings")


if __name__ == "__main__":
    ee = check_install()
    authenticate(ee)
    test_connection(ee)
    print("\n[done] GEE setup complete. You can now run the extraction scripts.")
    print("       Next: python pipeline/extract_viirs.py")
