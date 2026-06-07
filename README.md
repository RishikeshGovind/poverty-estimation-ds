# AfricaLens

**Can a satellite photo predict poverty?**

That is the question this project tries to answer.

**Live demo:** [africalens.github.io](https://rishikeshgovind.github.io/poverty-estimation-ai/) | **API:** Render (FastAPI)

---

## What Is This Project?

Around 900 million people still live in extreme poverty. A third of them are in Africa. Governments and aid organisations need to know *where* poverty is worst and whether it is getting better over time. But running household surveys is expensive, slow, and only covers a small fraction of communities.

Satellites photograph the entire Earth every few days, for free. This project builds a system that reads those photos and estimates how wealthy or poor a community is, without anyone needing to knock on a single door.

We pull satellite images from space, feed them into a deep learning model (the same type of AI used in facial recognition or self-driving cars), and produce poverty estimates for thousands of communities across Africa. The results are shown on an interactive map.

---

## Project Proposal

This project is built around the same three research goals as the [Chalmers University PhD project on AI and Global Development](https://www.aidevlab.org):

**Goal 1: Predict poverty from satellite images**
Train a computer vision AI to look at satellite images of villages and cities and output a score for how wealthy or poor that area is. We use survey data from Kenya and Nigeria as ground truth.

**Goal 2: Compare different satellites**
Not all satellite data is equal. Some take sharp 10-metre photos. Others take blurrier 30-metre ones. Some only capture light at night. We test which combination gives the best poverty estimates at the lowest data cost.

**Goal 3: Explain what the model sees**
A model that says "this area is poor" is not useful if nobody knows why it said that. We draw a heatmap over each satellite photo showing which parts of the image (rooftops, roads, lit streets) drove the prediction. This is essential for policymakers to trust the results.

---

## The Problem

Traditional poverty surveys like the [Demographic and Health Surveys (DHS)](https://dhsprogram.com) send researchers door to door to ask about household assets, housing quality, and income. These surveys are the gold standard. But they cost millions of dollars to run, only happen every 5 years, and cover only a sample of communities.

Satellite data is different. Satellites revisit the same spots repeatedly, at no cost per visit. From orbit you can see:

- How many lights are on at night (a signal of electricity access and economic activity)
- How densely packed the buildings are
- How much vegetation and farmland exists
- Whether roads are paved or dirt tracks

The question is whether those visual signals, read by an AI, can substitute for a household survey.

---

## Data We Used

### Poverty labels

The DHS surveys give each cluster (roughly a village or city neighbourhood) a **wealth index** score between about -3 and +3. A score near +2 means a wealthy urban area. A score near -2 means a very poor rural area. These scores are our training targets.

We have real survey data for **Kenya and Nigeria** (about 3,000 clusters from the 2022 to 2023 survey rounds). For 26 other African countries, we only have national-level satellite summaries, so predictions there are rough estimates rather than cluster-level ones.

### Satellite data

All satellite data is pulled through [Google Earth Engine](https://earthengine.google.com), a free platform for analysing satellite images at scale. The extraction pipeline lives in `pipeline/`.

**VIIRS nighttime lights (2014 to 2024)**
A NASA satellite photographs Earth every night. Brighter spots have more electricity. We use 10 years of these photos as a stand-in for electrification and economic activity. We also use them in the first training stage (see below).

**Sentinel-2 colour photos (2019 to present)**
A European Space Agency satellite takes colour photos at 10 metres per pixel. We cut a 256x256 pixel square around each survey cluster. That covers roughly 2.5 km x 2.5 km, about the size of a large neighbourhood. These images are the main input to our deep learning models.

**Landsat 8/9 photos (2014 to 2024)**
An older NASA satellite with 30-metre resolution. Less sharp than Sentinel-2 but with a longer history. Useful for comparing how much resolution actually matters.

**Sentinel-1 radar (ongoing)**
Radar satellites use their own radio signal instead of sunlight, so they work through clouds and at night. We test whether adding radar data improves predictions in cloud-heavy regions.

**Extra layers**
We also include land cover maps (forests, crops, buildings, water), population density maps, building footprint data, and estimated distances to roads, hospitals, and schools.

---

## How We Measure "How Green" or "How Built Up" an Area Is

Two simple formulas tell us a lot from a satellite photo:

**NDVI (how green is this area?)**
High NDVI means lots of vegetation like forests or crops. Low NDVI means roads, buildings, or bare soil.
Formula: `(near-infrared - red) / (near-infrared + red)`

**NDBI (how built up is this area?)**
High NDBI means lots of concrete and rooftops.
Formula: `(shortwave-infrared - near-infrared) / (shortwave-infrared + near-infrared)`

---

## The Models

### Main model: ResNet18 image classifier (`models/resnet_model.py`)

ResNet18 is a well-known deep learning model that was originally designed to classify everyday photos (cats, dogs, cars, etc.). We replace its final classification layer with a layer that outputs a single number: the predicted wealth index.

The model was originally trained on millions of everyday photos (called ImageNet pretraining). That gives it a head start in recognising shapes, textures, and edges before it ever sees a satellite image.

```
Satellite image (256 x 256 pixels)
    |
ResNet18 (finds patterns in the image)
    |
Single number: predicted wealth index
```

### Multi-task model (`models/multitask_model.py`)

This version predicts three things at once from the same image:

```
Same satellite image
    |
ResNet18
    |
    +-- Predicted wealth index     (SDG 1: No Poverty)
    +-- Predicted nighttime lights (SDG 7: Clean Energy)
    +-- Predicted built-up level   (SDG 11: Sustainable Cities)
```

Sharing the image analysis step saves compute and lets each prediction task help the others improve.

### Time-series model (`models/temporal_model.py`)

This model looks at satellite images from several different years and tracks how an area changes over time. It uses a ResNet18 to read each year's image, then an LSTM (a type of AI designed for sequences) to track the trend across years.

```
Images from year 1, year 2, year 3 ...
    |
ResNet18 reads each image separately
    |
LSTM tracks changes over time
    |
Wealth index trend
```

> **Current status:** The model is built. Training requires matched images and survey data from multiple years. We have that data for Kenya (2014 and 2022) and Nigeria (2018 and 2023).

### Vision Transformer (`models/vit_model.py`)

A newer type of image AI that splits a photo into small patches and analyses how each patch relates to every other one. We include it to compare against the ResNet18 approach.

### Tabular baseline (`pipeline/phase2_train.py`)

For countries where we do not have satellite image patches, we use a simpler model called Gradient Boosting. Instead of looking at images, it takes nine summary numbers per cluster (average nighttime light, vegetation index, urban/rural flag, location, etc.) and predicts a wealth index from those numbers alone. This model currently drives the live map for all 28 countries.

---

## How We Train

### Step 1: Learn from nighttime lights first

Before touching any poverty survey data, we first train the model on a much larger free task: predicting nighttime light levels from daytime satellite photos.

Why? Nighttime lights are a known proxy for wealth and electricity access. By predicting them first, the model learns to notice rooftops, lit streets, and road grids. Then we switch the training target to DHS wealth scores and fine-tune from there. This approach is called pretraining. It was introduced for this exact use case in [Jean et al. (2016)](https://science.sciencemag.org/content/353/6301/790) and significantly improves accuracy when labelled survey data is scarce.

### Step 2: Fine-tune on DHS wealth labels

We load the pretrained model and continue training it using real DHS survey scores as the target. The model already understands satellite images, so it can now focus on learning the specific connection between visual patterns and wealth.

Training settings:
- Optimiser: Adam, learning rate 0.0001
- Loss: mean squared error
- Batch size: 16 images
- Max epochs: 20

### Spatial cross-validation

A naive random split of survey clusters into training and test sets is not honest. Clusters in the same country are geographically close to each other. The model can score well on the test set without actually generalising, because it has memorised nearby clusters in training.

We use leave-one-country-out cross-validation instead. Each test round trains on all countries except one, then tests on that held-out country. This is a strict test of whether the model has learned patterns that transfer to places it has never seen.

### Confidence estimation

When the model makes a prediction, we run it 50 times with dropout (random neurons are switched off each time). The spread across those 50 predictions tells us how confident the model is. High spread means uncertain. Low spread means confident. This helps flag which clusters most need real survey verification.

---

## Comparing Satellites (`experiments/compare_satellites.py`)

We train the same model architecture six times using different satellite data combinations:

| Combination | Input channels | What it tests |
|---|---|---|
| Sentinel-2 only | 3 | Colour photos from space |
| Sentinel-1 only | 2 | Radar only (works in clouds) |
| VIIRS only | 1 | Nighttime lights only |
| Sentinel-2 + Sentinel-1 | 5 | Colour + radar |
| Sentinel-2 + VIIRS | 4 | Colour + nighttime lights |
| Sentinel-2 + Sentinel-1 + VIIRS | 6 | All three |

This shows which satellite (or combination) gives the best poverty estimates relative to data cost. Results go to `outputs/experiments/satellite_comparison.csv`.

**A note on resolution:** The Chalmers PhD project also plans to compare very high-resolution commercial satellites (Pléiades, 2 metres per pixel) against Sentinel-2 (10 m) and Landsat (30 m). We cover the Sentinel-2 vs Landsat comparison here. Pléiades requires a paid licence and is not included.

---

## What Is the Model Looking At? (`explainability/`)

A poverty prediction is only useful if policymakers can trust it. To build that trust, we need to show *what* in the image drove the model's decision.

We use **GradCAM** (Gradient-weighted Class Activation Mapping) to draw a heatmap over each satellite image. Pixels that most influenced the prediction glow bright. If the bright spots land on rooftops, roads, and lit streets, the model is picking up on the right things. If they land on random empty sky, something is wrong.

**GradCAM++** is an improved version that produces sharper heatmaps when several different regions in the image are important at the same time.

How to run it:

```bash
# Default: 8 sample images, wealth index task
python -m explainability.run_explainability

# More samples, energy access task
python -m explainability.run_explainability --n-samples 16 --task sdg7_ntl

# Multi-task model, cities task
python -m explainability.run_explainability \
    --checkpoint outputs/models/multitask_best.pth \
    --model-type multitask --task sdg11_buildup
```

Output: a grid of images saved to `outputs/explainability/`. Each row shows the original satellite patch, the GradCAM heatmap, and the GradCAM++ heatmap.

---

## Results

### Gradient Boosting model on Kenya + Nigeria (80/20 train/test split)

| Metric | Value |
|--------|-------|
| R² | **0.776** |
| RMSE | 0.401 |
| MAE | 0.306 |
| Training samples | 2,439 |
| Test samples (held out) | 609 |

An R² of 0.776 means the model explains 77.6% of the variation in wealth across survey clusters. The most important features are the urban/rural flag (48%), latitude (16%), and average nighttime light level (10%). The strong influence of location confirms that geography is a major driver of wealth differences in this dataset.

> These numbers come from an 80/20 train/test split. A stricter test using leave-one-country-out cross-validation is available in `experiments/spatial_cv_experiment.py`.

### Deep learning model (ResNet18)

The CNN model code is complete and the inference pipeline is ready. All existing checkpoints currently produce negative R² on held-out data, meaning a simple mean prediction beats them. The bottleneck is completing the nighttime-lights pretraining step before fine-tuning on DHS labels. Once a checkpoint with positive R² is ready, it will replace the tabular model automatically in `pipeline/phase2_predict.py`.

| Checkpoint | R² (held out) |
|---|---|
| s2_kenya_nigeria.pth | -2.86 |
| best_model.pth | -2.06 |
| s2_kenya_real.pth | -0.75 |
| **Gradient Boosting (live demo)** | **+0.776** |

### How this compares to published research

| Method | Dataset | R² |
|--------|---------|-----|
| Jean et al. 2016 (CNN with NTL pretraining) | Uganda, Tanzania | ~0.63 |
| Yeh et al. 2020 (multi-task CNN) | 23 African countries | ~0.70 |
| **This project (Gradient Boosting)** | Kenya, Nigeria | **0.776** |

Direct comparisons are not straightforward because the datasets and countries differ. This table gives a rough sense of where this project sits relative to published work, not a claim of superiority. Our CNN, once properly pretrained, is the architecture most comparable to those papers.

---

## SDG Scores (`scoring/sdg_scorer.py`)

The United Nations Sustainable Development Goals (SDGs) set targets for reducing poverty, expanding energy access, and building sustainable cities by 2030. We estimate a 0 to 100 progress score for three of those goals using satellite signals as proxies.

**SDG 1 (No Poverty)**
Based on the predicted wealth index. Score of 100 means wealthiest observed cluster. Score of 0 means poorest.

**SDG 7 (Clean Energy)**
Based on nighttime light level. High light level suggests reliable electricity access.

**SDG 11 (Sustainable Cities)**
Based on how bright the visible image is, which correlates with built-up surfaces and infrastructure.

**Composite score**
A weighted average across all three goals. SDG 1 gets double the weight of SDG 7 and SDG 11.

> These scores use thresholds estimated from the satellite data distributions, not from official UN monitoring methods. They are useful for comparing clusters and tracking trends. They should not be reported as official SDG statistics without further validation against survey data.

---

## Project Structure

```
poverty-estimation-ai/
├── pipeline/                  # Step 1: pull satellite data from Google Earth Engine
│   ├── config.py              #   list of 30 African countries and settings
│   ├── extract_viirs.py       #   nighttime lights (2014 to 2024)
│   ├── extract_sentinel2.py   #   vegetation and built-up indices
│   ├── extract_landsat.py     #   Landsat NDVI (2014 to 2024)
│   ├── phase2_aggregate.py    #   combine features per survey cluster
│   ├── phase2_train.py        #   train the Gradient Boosting baseline model
│   └── run_phase1.py          #   run all extractions in order
│
├── training/
│   ├── pretrain_ntl.py        # Step 2a: pretrain on nighttime lights
│   ├── trainer.py             # Step 2b: fine-tune on DHS wealth labels
│   ├── multitask_trainer.py   # train the three-goal model
│   ├── dataset.py             # loads satellite patches and survey labels
│   ├── multi_sensor_dataset.py# loads multiple satellite types together
│   ├── spatial_cv.py          # leave-one-country-out evaluation
│   └── train.py               # main entry point for training
│
├── models/
│   ├── resnet_model.py        # main CNN model
│   ├── multitask_model.py     # three-goal CNN model
│   ├── temporal_model.py      # time-series CNN + LSTM model
│   ├── vit_model.py           # Vision Transformer model
│   └── tabular_fusion_model.py
│
├── experiments/
│   ├── compare_satellites.py  # test 6 different satellite combinations
│   └── spatial_cv_experiment.py
│
├── explainability/
│   ├── gradcam.py             # GradCAM and GradCAM++ code
│   └── run_explainability.py  # run and save heatmap images
│
├── scoring/
│   ├── sdg_scorer.py          # convert predictions to SDG 0-100 scores
│   ├── run_inference.py       # run model on all clusters
│   └── generate_geojson.py    # export results to map format
│
├── utils/
│   ├── uncertainty.py         # confidence estimation (Monte Carlo Dropout)
│   ├── config.py              # loads settings from config.yaml
│   └── logging.py
│
├── server/                    # web API (FastAPI, hosted on Render)
├── client/                    # interactive map (React + Leaflet)
├── config.yaml                # all settings and paths in one file
└── render.yaml                # deployment config
```

---

## How to Run It

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For the satellite pipeline, you also need a free Google Earth Engine account:

```bash
earthengine authenticate
```

### 2. Pull satellite data

```bash
cd pipeline
python run_phase1.py --years 2022 2023   # quick test with 2 years of data
python run_phase1.py                     # full run from 2014 to 2024
```

Output: `pipeline/outputs/satellite_features.json`

### 3. Train the model

```bash
# Recommended: pretrain on nighttime lights first
python -m training.pretrain_ntl

# Fine-tune on DHS wealth labels
python -m training.train

# Train the three-goal version
python -m training.train --model-type multitask
```

### 4. Run the satellite comparison

```bash
python -m experiments.compare_satellites
# Results saved to outputs/experiments/satellite_comparison.csv
```

### 5. Generate explainability heatmaps

```bash
python -m explainability.run_explainability --n-samples 12
# Images saved to outputs/explainability/
```

### 6. Export predictions to the map

```bash
python -m pipeline.phase2_predict
# Output: client/public/predictions.geojson
```

### 7. Run the API locally

```bash
uvicorn server.main:app --reload --port 8000
```

---

## Deployment

| Part | Where it runs | How it deploys |
|---|---|---|
| Frontend map | GitHub Pages | Automatically on git push |
| Web API | Render | Automatically on git push |

The `predictions.geojson` file is stored in the repo and served as a static file. The map loads it directly without calling the API.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | The AI chat assistant on the map |
| `ACLED_API_KEY` | Conflict events data layer |
| `ACLED_EMAIL` | ACLED account email |
| `GEE_PROJECT` | Your Google Earth Engine project ID |
| `GEE_KEY_FILE` | Path to your GEE service account key (for automated runs) |

---

## Known Limitations

| Limitation | Why it exists | What we plan to do next |
|---|---|---|
| Only Kenya and Nigeria have real survey-level predictions | DHS data access takes time to arrange | Add Ethiopia, Ghana, and Tanzania as data becomes available |
| No very-high-resolution satellite (Pléiades, 2 m per pixel) | Requires a paid commercial licence | Apply for ESA Third Party Missions access |
| The CNN model currently performs worse than the tabular model | Too little labelled data without the nighttime-lights pretraining step | Complete pretraining and retrain the CNN |
| The time-series model is built but not yet trained end-to-end | Needs matched survey data across two or more years | Train on Kenya 2014/2022 and Nigeria 2018/2023 |
| SDG thresholds are estimated from satellite data, not calibrated against surveys | No ground-truth benchmark was available | Cross-check thresholds against DHS electricity access responses |

---

## Why We Made Certain Choices

**Why DHS labels?**
DHS surveys are the most trusted source of sub-national poverty data in low-income countries. The wealth index is a continuous score, which lets us treat poverty prediction as a regression problem (predicting a number on a scale) rather than a simple yes/no classification. That produces more useful output.

**Why pretrain on nighttime lights?**
VIIRS nighttime light data covers the whole globe and goes back to 2012. It gives us millions of free training examples. By teaching the model to predict nighttime lights first, we give it a head start in understanding what wealth-related features look like from space. Then we fine-tune on the much smaller DHS dataset. This idea comes from Jean et al. (2016) and is one of the most cited techniques in this field.

**Why leave-one-country-out cross-validation?**
If we randomly split survey clusters into training and test sets, clusters from the same country end up on both sides. Because nearby clusters share the same weather, soil, and local economy, the model can score well on the test set just by memorising nearby training examples, without actually generalising. Holding out an entire country forces a much harder and more honest test.

**Why GradCAM on the last convolutional layer?**
The last convolutional layer in ResNet18 captures high-level patterns like "is there a dense road grid here?" rather than low-level ones like "is there a horizontal edge here?". It also still holds spatial information (which part of the image triggered the pattern), which the pooling layer immediately after it throws away. That combination makes it the right layer to visualise.

**Why a tabular model for countries beyond Kenya and Nigeria?**
To run the CNN, we need a 256x256 pixel satellite image patch centred on each survey cluster. We only have those patches for Kenya and Nigeria. For the other 26 countries, we only have one average number per country per year. The Gradient Boosting model can work with those country-level averages and still give a rough poverty estimate, so it fills the gap while we expand the image patch dataset.
