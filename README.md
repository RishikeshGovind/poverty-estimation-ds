# AfricaLens — Poverty Estimation from Satellite Imagery

A geospatial AI system that estimates poverty levels and SDG progress across Sub-Saharan Africa using multi-source satellite imagery and deep learning. Built as an independent research prototype addressing the three core objectives of the [AI and Global Development Lab](https://www.aidevlab.org) at Chalmers University.

**Live demo:** [africalens.github.io](https://rishikeshgovind.github.io/poverty-estimation-ai/) &nbsp;|&nbsp; **API:** Render (FastAPI)

---

## Problem Statement

Around 900 million people still live in extreme poverty, with one third concentrated in Africa. Policymakers lack the fine-grained, geo-temporal data needed to identify which communities are on track to reach the Sustainable Development Goals and which interventions are working. Traditional surveys (DHS, LSMS) are expensive, infrequent, and spatially sparse.

This project asks: **can satellite imagery, combined with deep learning, substitute for ground surveys and produce high-resolution, continuously updated poverty maps?**

The work is structured around three objectives that mirror the PhD research agenda at Chalmers:

| Objective | What we do |
|---|---|
| WP1 — Deep learning for multidimensional poverty | Train ResNet18 / ViT models on Sentinel-2 patches to predict DHS wealth index and SDG proxy indicators |
| WP2 — Satellite comparison | Benchmark S2 (optical), S1 (SAR), and VIIRS (nighttime lights) separately and in fusion to identify the best precision/cost tradeoff |
| WP3 — Explainability for policy trust | Apply GradCAM and GradCAM++ to identify which visual features drive each poverty prediction |

---

## Data Sources and Acquisition

### 1. Ground Truth — DHS Wealth Index

The Demographic and Health Surveys (DHS) provide the primary poverty labels. Each DHS cluster (a village or urban neighbourhood) is assigned a continuous **wealth index** (approximately −3 to +3) derived from household asset ownership and housing quality data. We use GPS-georeferenced cluster coordinates to spatially join survey points to satellite patches.

Countries with real DHS cluster data: **Kenya, Nigeria** (DHS 2022–23 rounds, ~3,000 geo-referenced clusters). An additional 26 Sub-Saharan African countries are included in the map using country-level satellite aggregates from Google Earth Engine; those predictions are approximations and should not be treated as survey-equivalent estimates.

Label construction follows the sustainlab approach: the cluster-level wealth index is used directly as the regression target. For multi-task training, additional SDG proxy labels are derived from satellite data (see SDG Scoring below).

### 2. Satellite Data — Google Earth Engine Pipeline

All satellite time series are extracted via the [Google Earth Engine Python API](https://developers.google.com/earth-engine). The pipeline lives in `pipeline/` and runs in three sequential stages merged by `pipeline/run_phase1.py`.

#### VIIRS Nighttime Lights (2014–2024)
- **Source:** `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG` (monthly gap-filled composites)
- **Band:** `avg_rad` — mean radiance in nW/cm²/sr
- **Spatial scale:** 500 m
- **Method:** Annual mean of 12 monthly composites, computed per country polygon using `reduceRegions` with a mean reducer. Covers 30 SSA countries from 2014 to 2024.
- **Use:** Primary proxy for electrification (SDG 7) and economic activity. Also used as the pretraining signal for Stage 1 training (see below).

#### MODIS NDVI (2019–2024)
- **Source:** `MODIS/061/MOD13A3` (pre-built monthly 1 km NDVI composites)
- **Method:** Annual mean of monthly composites, scale factor 0.0001 applied. Chosen over Sentinel-2 NDVI for country-level aggregation because MODIS pre-compositing eliminates per-scene cloud masking overhead.
- **Use:** Vegetation health proxy; correlates with agricultural productivity and rural livelihoods.

#### Landsat 8/9 NDVI and NDBI (2014–2024)
- **Source:** `LANDSAT/LC08/C02/T1_L2` + `LANDSAT/LC09/C02/T1_L2` (Collection 2 Level-2 Surface Reflectance)
- **Cloud masking:** QA_PIXEL band, bits 3 (cloud shadow) and 5 (cloud) masked.
- **Scale factors:** SR = raw × 0.0000275 − 0.2 (Collection 2 standard)
- **NDVI:** (NIR SR_B5 − Red SR_B4) / (NIR + Red), annual median composite
- **NDBI:** (SWIR SR_B6 − NIR SR_B5) / (SWIR + NIR), annual median composite
- **Spatial scale:** 1000 m (country-level stats); 30 m native (patch extraction)
- **Use:** Longer temporal record than Sentinel-2 (back to 2013) enables trend analysis. NDBI (Normalized Difference Built-up Index) is a built-up area proxy for SDG 11.

#### Sentinel-2 Image Patches (10 m)
- **Source:** Microsoft Planetary Computer STAC API (`pystac-client`, `planetary-computer`)
- **Bands:** RGB (B04, B03, B02) for standard runs; additional bands for multi-stream experiments
- **Cloud cover threshold:** ≤10%
- **Normalization:** Surface reflectance values (0–10000) scaled to [0, 1]
- **Patch size:** 256 × 256 pixels centred on each DHS cluster coordinate
- **Use:** Primary imagery for all deep learning models. At 10 m/px a 256-pixel patch covers a ~2.56 km² area around each survey cluster.

#### Sentinel-1 SAR (VV + VH)
- **Bands:** VV (co-polarization, sensitive to surface roughness/structure) and VH (cross-polarization, sensitive to vegetation volume)
- **Normalization:** Linear power scale clipped to [0, 0.5] then divided by 0.5
- **Use:** Weather-independent imaging. The S2+S1 fusion experiment tests whether adding SAR improves poverty estimation in cloud-prone regions.

#### Supplementary Layers
- **ESA WorldCover 2021** — 10 m land use classification (10 classes: trees, cropland, built-up, water, etc.)
- **Meta HRSL** — high-resolution settlement layer, population count per 30 m cell
- **Google/Microsoft Building Footprints** — building density rasterised to patch resolution
- **OSM accessibility features** — distance to nearest road, hospital, school within 5 km radius

---

## Model Architecture

### Single-Task Regressor — `models/resnet_model.py`

```
Input (B, C, 256, 256)
    │
ResNet18 backbone (conv1 → layer4 → avgpool)   [pretrained ImageNet]
    │  (B, 512)
Dropout(p)
    │
Linear(512 → 1)
    │
Wealth index prediction (scalar)
```

The first convolution layer is patched to accept an arbitrary number of input channels (any combination of S2/S1/VIIRS bands). Extra channels beyond 3 are initialised with the mean of the pretrained RGB weights to preserve scale.

### Multi-Task Regressor — `models/multitask_model.py`

Extends the single-task design with **one independent regression head per SDG task**, all sharing the same ResNet18 backbone:

```
Shared backbone (B, 512)
    ├── Linear(512→1)  →  SDG 1: wealth index
    ├── Linear(512→1)  →  SDG 7: nighttime lights (electrification proxy)
    └── Linear(512→1)  →  SDG 11: built-up brightness (infrastructure proxy)
```

Training uses a weighted sum of masked MSE losses. NaN labels (e.g., when VIIRS patches are absent for a cluster) are masked out so they do not corrupt gradients.

### Temporal Model — `models/temporal_model.py`

For time-series analysis across survey years:

```
Input (B, T, C, H, W)  — T time steps
    │
ResNet18 (shared weights across time)  →  (B, T, 512)
    │
LSTM(512 → 128)
    │
Linear(128 → 1)  →  Wealth index at final time step
```

> **Status:** Architecture implemented; end-to-end temporal training requires paired (image, label) sequences across multiple DHS survey rounds, which are available for Kenya (2014, 2022) and Nigeria (2018, 2023). Full temporal evaluation is planned as a next step.

### Vision Transformer — `models/vit_model.py`

`vit_base_patch16_224` from `timm`, classification head replaced with a scalar regression head. Used for architecture comparison experiments.

### Tabular Baseline — `pipeline/phase2_train.py`

Gradient Boosting Regressor (500 estimators, max depth 4) on nine hand-crafted cluster-level features: NTL mean/std/max, Excess Green index, S2 brightness, NTL trend slope, urban flag, latitude, longitude.

This model is currently used for all-country production predictions in `predictions.geojson` because it can run on country-level satellite aggregates where per-cluster image patches are unavailable. The CNN/ViT models require actual Sentinel-2 patches centred on each DHS cluster coordinate and are the research focus; they are trained and evaluated on the Kenya + Nigeria patch dataset.

---

## Training Strategy

Training follows a two-stage procedure adapted from the [sustainlab](https://github.com/sustainlab-group) methodology.

### Stage 1 — NTL Pretraining (`training/pretrain_ntl.py`)

Nighttime lights (VIIRS) are a freely available, data-rich proxy for economic activity with 10× more geographic coverage than DHS. In Stage 1, ResNet18 is trained to **predict VIIRS radiance from Sentinel-2 RGB patches**. This forces the backbone to learn visual features correlated with electrification (rooftops, road grids, lit infrastructure) without requiring any survey labels.

The pretrained backbone weights are saved and used to initialise Stage 2.

### Stage 2 — DHS Fine-tuning (`training/trainer.py`)

The pretrained backbone is loaded and the regression head is re-trained on DHS wealth index labels. Only layers with matching shapes are transferred; the new conv1 weights for non-RGB channel counts are initialised from scratch.

**Optimiser:** Adam, lr = 1e-4, ReduceLROnPlateau (patience 3)  
**Loss:** MSE (single-task) or weighted sum of masked MSEs (multi-task)  
**Augmentation:** Random horizontal and vertical flips  
**Batch size:** 16 | **Default epochs:** 20

### Spatial Cross-Validation (`training/spatial_cv.py`)

Standard random train/test splits leak information across nearby geographic points. We use **leave-one-country-out cross-validation** to evaluate true generalisation: each fold trains on all countries except one, then tests on the held-out country. This enforces strict spatial separation and tests whether models learned transferable visual-poverty associations rather than country-specific artefacts.

If country labels are unavailable, the fallback is k-means spatial clustering (k = 5) to create pseudo-country folds that still enforce geographic separation.

### Uncertainty Estimation (`utils/uncertainty.py`)

Monte Carlo Dropout is applied at inference time: 50 stochastic forward passes are run with dropout layers kept active, and the mean and standard deviation across passes are returned. The standard deviation is an epistemic uncertainty estimate — high values indicate the model is uncertain about a cluster, useful for flagging areas that need ground truth verification.

Calibration is tracked via ±1σ and ±2σ coverage rates (target: ~68% and ~95%).

---

## Satellite Comparison Experiment (`experiments/compare_satellites.py`)

Six sensor combinations are trained with identical architectures and hyperparameters to isolate the effect of sensor choice:

| Combination | Channels | What it tests |
|---|---|---|
| S2-only | 3 | Optical RGB baseline |
| S1-only | 2 | SAR-only (all-weather) |
| VIIRS-only | 1 | Nighttime lights only |
| S2 + S1 | 5 | Optical + radar fusion |
| S2 + VIIRS | 4 | Optical + NTL fusion |
| S2 + S1 + VIIRS | 6 | Full fusion |

Metrics — R², RMSE, MAE — are recorded per run and saved to `outputs/experiments/satellite_comparison.csv`. This experiment isolates the **sensor-type** dimension of the precision/cost trade-off (optical vs. SAR vs. NTL).

> **Note on WP2 scope:** The Chalmers PhD agenda additionally targets a **spatial resolution** comparison — Pléiades (2 m), Sentinel-2 (10 m), and Landsat (30 m). Pléiades requires commercial licensing and is not included here. This project covers the Sentinel-2 (10 m) vs. Landsat (30 m) axis; the Landsat extraction pipeline (`pipeline/extract_landsat.py`) produces 30 m NDVI/NDBI patches that can be used in the same training framework as a resolution baseline.

---

## Explainability — GradCAM and GradCAM++ (`explainability/`)

WP3 of the research agenda requires understanding *what the model sees* in a satellite patch when it predicts poverty. This is essential for policy trust: if the model is reacting to vegetation cover, rooftop density, road networks, or lighting — policymakers need to know which.

### Method

We apply **GradCAM** and **GradCAM++** to the final convolutional block of ResNet18 (`backbone.layer4[-1]`). For a given input patch and task, the gradient of the scalar task output with respect to the feature maps at that layer is computed; these gradients are global-average-pooled into channel weights, which are then used to linearly combine the feature maps into a spatial attention heatmap.

GradCAM++ improves on GradCAM by weighting each pixel's gradient contribution by its second-order importance, producing sharper localisation especially when multiple discriminative regions exist in one patch.

### Outputs

- **Figure grid** (`outputs/explainability/gradcam_<task>.png`) — each row shows: original satellite patch | GradCAM overlay | GradCAM++ overlay, annotated with the predicted and true wealth index.
- **Summary CSV** (`outputs/explainability/gradcam_summary_<task>.csv`) — per-patch metrics:
  - `gc_top10_mean` — mean activation in the top-10% most active pixels (higher = model focuses on a small, high-confidence region)
  - `gc_entropy` — Shannon entropy of the activation distribution (lower = sharper, more localised attention)

### Usage

```bash
# Default: 8 samples, primary task (sdg1_wealth)
python -m explainability.run_explainability

# More samples, different task
python -m explainability.run_explainability --n-samples 16 --task sdg7_ntl

# Multi-task model
python -m explainability.run_explainability \
    --checkpoint outputs/models/multitask_best.pth \
    --model-type multitask --task sdg11_buildup
```

Works with both `ResNetRegression` and `MultiTaskResNet`. No additional dependencies — `pytorch-grad-cam` (`grad-cam` on PyPI) is already included.

---

## Results

### Tabular baseline (GBR) — Kenya + Nigeria, 80/20 random split

| Metric | Value |
|--------|-------|
| R² | **0.776** |
| RMSE | 0.401 |
| MAE | 0.306 |
| Training samples | 2,439 |
| Validation samples | 609 |

Top feature importances: `is_urban` (48%), `latitude` (16%), `ntl_mean` (10%), `lon` (7%). The dominance of `is_urban` and geographic coordinates reflects the strong urban-rural wealth gradient in DHS data and confirms the satellite signal is meaningful but coordinates carry substantial baseline variance.

> **Note on evaluation:** These metrics are from an 80/20 random split on the Kenya + Nigeria DHS dataset. For deployment-realistic estimates, see `experiments/spatial_cv_experiment.py` which runs leave-one-country-out cross-validation to enforce strict geographic separation between train and test sets.

### Deep learning (ResNet18) — preliminary

CNN model training requires per-cluster Sentinel-2 patches (256 × 256 px). The `s2_kenya_nigeria.pth` checkpoint drives the live `predictions.geojson` (3,048 DHS clusters). Full benchmark results against the tabular baseline using spatial CV are in progress.

> **Prediction range note:** The ResNet18 head is an unconstrained linear layer, so predictions can exceed the nominal DHS wealth index range of [−3, +3]. The downstream `wi_to_poverty()` function clips to [0, 100%] so the map display is unaffected, but the raw `wealth_index` property in the GeoJSON may include values outside this range on patches the model has not seen during training.

### Comparison to prior work

| Method | Dataset | R² |
|--------|---------|-----|
| Jean et al. 2016 (transfer learning, NTL pretrain) | Uganda, Tanzania | ~0.63 |
| Yeh et al. 2020 (multi-task CNN) | 23 African countries | ~0.70 |
| **This work — GBR tabular (Kenya + Nigeria)** | Kenya, Nigeria | **0.776** |

The tabular result is not directly comparable (different countries, different feature engineering) but situates the project in the relevant literature.

---

## SDG Scoring (`scoring/sdg_scorer.py`)

Raw model outputs are mapped to interpretable 0–100 SDG progress scores using threshold-based linear scaling:

**SDG 1 — No Poverty**
```
score = clip((wealth_index − min) / (max − min), 0, 1) × 100
```
DHS wealth index range: [−2.0, +2.0]. A score of 100 = wealthiest observed; 0 = poorest observed.

**SDG 7 — Clean Energy** (electrification proxy via VIIRS NTL)
```
score = clip(ntl_normalised / ntl_threshold, 0, 1) × 100
```
NTL values above the threshold indicate reliable electricity access.

**SDG 11 — Sustainable Cities** (built-up area proxy via S2 brightness)
```
score = clip(brightness / buildup_threshold, 0, 1) × 100
```
Higher brightness correlates with built-up surface area and infrastructure density.

**Composite score** — weighted average across available tasks (SDG1: weight 1.0, SDG7: 0.5, SDG11: 0.5).

> **Caveat:** The thresholds and weights above are heuristic proxies derived from the satellite data distributions, not from official SDG monitoring methodology. They are appropriate for relative ranking and trend analysis but should not be reported as SDG-compliant metrics without validation against ground-truth survey data.

Scores are stored in `docs/data/predictions.geojson` and rendered on the interactive Leaflet map.

---

## Repository Structure

```
poverty-estimation-ai/
├── pipeline/                  # Phase 1: GEE satellite extraction
│   ├── config.py              #   30-country SSA list, GEE project settings
│   ├── extract_viirs.py       #   VIIRS NTL (2014–2024)
│   ├── extract_sentinel2.py   #   MODIS NDVI + Landsat NDBI via GEE
│   ├── extract_landsat.py     #   Landsat 8/9 NDVI (2014–2024)
│   ├── phase2_aggregate.py    #   Cluster-level feature aggregation
│   ├── phase2_train.py        #   Gradient Boosting tabular baseline
│   └── run_phase1.py          #   Orchestrates all extractions + merges
│
├── training/
│   ├── pretrain_ntl.py        # Stage 1: NTL pretraining
│   ├── trainer.py             # Stage 2: single-task training + satellite comparison
│   ├── multitask_trainer.py   # Multi-task SDG training
│   ├── dataset.py             # PovertyDataset (S2 patches + DHS labels)
│   ├── multi_sensor_dataset.py# Multi-channel sensor fusion dataset
│   ├── spatial_cv.py          # Leave-one-country-out cross-validation
│   └── train.py               # Entry point for a single training run
│
├── models/
│   ├── resnet_model.py        # ResNetRegression + ResNetNTLPretrain
│   ├── multitask_model.py     # MultiTaskResNet (SDG1/7/11 heads)
│   ├── temporal_model.py      # CNN + LSTM for time-series
│   ├── vit_model.py           # Vision Transformer regression
│   └── tabular_fusion_model.py
│
├── experiments/
│   ├── compare_satellites.py  # 6-way sensor combination benchmark
│   └── spatial_cv_experiment.py
│
├── explainability/
│   ├── gradcam.py             # GradCAM + GradCAM++ core module
│   └── run_explainability.py  # CLI runner → figures + summary CSV
│
├── scoring/
│   ├── sdg_scorer.py          # Raw predictions → SDG 0–100 scores
│   ├── run_inference.py       # Batch inference on all clusters
│   └── generate_geojson.py    # Scores → predictions.geojson
│
├── utils/
│   ├── uncertainty.py         # Monte Carlo Dropout inference
│   ├── config.py              # config.yaml loader
│   └── logging.py
│
├── server/                    # FastAPI backend (Render)
├── docs/                      # GitHub Pages frontend (Leaflet.js)
│   ├── index.html
│   └── data/predictions.geojson
├── config.yaml                # All hyperparameters and paths
└── render.yaml                # Render deployment config
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Authenticate with Google Earth Engine (needed for Phase 1 only):

```bash
earthengine authenticate
```

### 2. Run the satellite extraction pipeline (Phase 1)

```bash
cd pipeline
python run_phase1.py --years 2022 2023   # quick 2-year test
python run_phase1.py                     # full 2014–2024 run
```

Outputs: `pipeline/outputs/satellite_features.json`

### 3. Train the model

```bash
# Stage 1: NTL pretraining (optional but recommended)
python -m training.pretrain_ntl

# Stage 2: Single-task wealth index regression
python -m training.train

# Multi-task (SDG1 + SDG7 + SDG11)
python -m training.train --model-type multitask
```

### 4. Run the satellite comparison experiment

```bash
python -m experiments.compare_satellites
# Results → outputs/experiments/satellite_comparison.csv
```

### 5. Generate explainability figures

```bash
python -m explainability.run_explainability --n-samples 12
# Output → outputs/explainability/gradcam_sdg1_wealth.png
```

### 6. Score and export to GeoJSON

```bash
python -m scoring.run_inference
python -m scoring.generate_geojson
# Output → docs/data/predictions.geojson  (auto-served by GitHub Pages)
```

### 7. Run the API locally

```bash
uvicorn server.main:app --reload --port 8000
```

---

## Deployment

| Layer | Platform | Trigger |
|---|---|---|
| Frontend (`docs/`) | GitHub Pages | Auto-deploys on `git push` to `main` |
| Backend (`server/`) | Render | Auto-deploys on `git push` to `main` |

No manual deploy step needed. The `docs/data/predictions.geojson` file is committed to the repo; the frontend reads it directly from GitHub Pages without needing the API for the static poverty map layer.

---

## Environment Variables

| Variable | Required for | Description |
|---|---|---|
| `GEMINI_API_KEY` | AI query endpoint | Gemini free tier |
| `ACLED_API_KEY` | Conflict layer | ACLED conflict events API |
| `ACLED_EMAIL` | Conflict layer | ACLED account email |
| `GEE_PROJECT` | Pipeline | GEE cloud project ID |
| `GEE_KEY_FILE` | Pipeline (CI/CD) | Path to GEE service account key |

Copy `.env.example` to `.env` and fill in your keys.

---

## Limitations and Roadmap

| Limitation | Detail | Planned fix |
|---|---|---|
| DHS coverage | Real cluster-level predictions only for Kenya and Nigeria. 26 other countries use country-level satellite aggregates scaled by empirical rules. | Integrate additional DHS rounds (Ethiopia, Ghana, Tanzania) as data access is obtained. |
| No Pléiades (2 m) | Commercial licensing required; not included in current satellite comparison. | Pléiades NEO data via ESA Third Party Missions would close the WP2 resolution axis. |
| CNN not in production path | `predictions.geojson` is generated by the GBR tabular model; CNN/ViT models require patch-level data. | Wire `scoring/run_inference.py` to use the fine-tuned ResNet18 for Kenya/Nigeria clusters once patch dataset is finalised. |
| Temporal model untrained | `TemporalModel` (CNN+LSTM) is implemented but not yet trained end-to-end. | Train on Kenya 2014/2022 and Nigeria 2018/2023 paired rounds; report wealth-change prediction accuracy. |
| Evaluation metric risk | `evaluate_model.py` currently reports metrics on the full dataset; proper test-set evaluation requires a held-out split committed before training. | Integrate spatial CV metrics as the canonical reported result. |
| SDG thresholds are heuristic | NTL and NDBI thresholds are not calibrated against household survey electricity access or infrastructure data. | Cross-validate thresholds against DHS household electrification responses. |

---

## Key Design Decisions

**Why DHS labels?** The Demographic and Health Surveys are the gold standard for sub-national poverty measurement in low-income countries. The continuous wealth index is more informative than binary poverty thresholds and enables regression rather than classification.

**Why NTL pretraining?** VIIRS nighttime lights cover the entire globe annually since 2012 at no cost, providing millions of (image, NTL) pairs. Pretraining on this abundant signal before fine-tuning on the scarce DHS labels significantly improves label efficiency — a core insight from Jean et al. (2016) and the sustainlab group.

**Why leave-one-country-out CV?** Random spatial splits allow information from nearby clusters to leak between train and test sets (Moran's I effect). Country-level holdout enforces strict spatial independence and gives a realistic estimate of out-of-distribution generalisation — the scenario that actually matters for deployment in new countries.

**Why GradCAM on layer4?** The final residual block captures the highest-level spatial abstractions (200–300 m receptive field at 10 m/px input). Earlier layers capture texture; later pooling discards spatial information entirely. `layer4[-1]` is the standard explainability target for ResNet architectures and has been validated in remote sensing literature.

**Why a tabular model for all-country predictions?** The CNN models require 256 × 256 px Sentinel-2 patches centred on known DHS cluster coordinates. Only Kenya and Nigeria currently have those patch datasets. For the other 26 countries, we have only country-level GEE aggregates (annual mean NDVI, NDBI, NTL). A GBR on nine derived features bridges this gap and allows a complete SSA map while the patch dataset is expanded — at the cost of coarser, less trustworthy predictions outside the DHS countries.
