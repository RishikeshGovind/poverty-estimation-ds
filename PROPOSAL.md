# Project Proposal

## Comparing Earth Observation and AI Methods for Sustainable Development

**AfricaLens** | Independent Research Prototype
**Live demo:** [africalens.github.io](https://rishikeshgovind.github.io/poverty-estimation-ai/)

---

## The Problem

About 900 million people still live in extreme poverty. A third of them are in Africa.

Governments and aid organisations need fine-grained, up-to-date data to know where poverty is worst and whether their interventions are working. The main tool available today is household surveys. Researchers travel door to door, asking families about their assets, housing quality, and income. These surveys are accurate, but they have real limits:

- They cost millions of dollars to run.
- They only happen every 5 to 10 years.
- They cover a small fraction of communities.
- Results take years to publish.

That means policymakers are often working with data that is years out of date. By the time a programme is designed around old survey data, the situation on the ground may have already changed.

Satellites photograph every corner of the Earth every few days. That data is free and continuously updated. The question this project asks is:

**Can we read satellite images with AI and produce poverty estimates that are as reliable as a household survey?**

---

## Research Goals

This project is structured around three goals that mirror the research agenda of the [AI and Global Development Lab](https://www.aidevlab.org) at Chalmers University of Technology.

### Goal 1: Predict poverty from satellite images

Train a deep learning model (a computer vision AI) to look at satellite images of villages and cities and output a poverty score for each community.

We use data from the [Demographic and Health Surveys (DHS)](https://dhsprogram.com), which assign each surveyed community a continuous wealth index between roughly -3 (very poor) and +3 (very wealthy). These scores are our ground truth labels.

The model we train is **ResNet18**, a well-known image recognition architecture. We replace its final classification layer with a regression layer that outputs a single wealth score. The model is first trained to predict nighttime light levels from daytime satellite photos (a technique introduced by Jean et al. 2016), then fine-tuned on the DHS wealth labels.

We also build a multi-task version that predicts three things at once from the same image:
- Wealth index (SDG 1: No Poverty)
- Nighttime light level as a proxy for electricity access (SDG 7: Clean Energy)
- Built-up area level as a proxy for infrastructure (SDG 11: Sustainable Cities)

And a time-series version that looks at images from several years and tracks how a community is changing over time.

### Goal 2: Compare different satellites

Not all satellite data is equal. Some satellites take sharp 10-metre photos. Others take blurrier 30-metre ones. Some only capture data at night. Each type has different cost and availability.

We train the same model architecture six times, each time using a different combination of satellite data:

| Combination | What it represents |
|---|---|
| Sentinel-2 only | Colour photos at 10 m per pixel |
| Sentinel-1 only | Radar images (works through clouds) |
| VIIRS nighttime lights only | Night light intensity |
| Sentinel-2 + Sentinel-1 | Colour photos + radar |
| Sentinel-2 + VIIRS | Colour photos + nighttime lights |
| All three combined | Full sensor fusion |

This experiment identifies which satellite gives the best poverty estimates relative to its data cost. The goal is to find the most practical combination for real deployment.

We also compare Sentinel-2 (10 m per pixel) against Landsat (30 m per pixel) to understand how much resolution matters. Note: Pléiades (2 m per pixel commercial satellite) is part of the Chalmers research agenda but requires a paid licence. It is not included in this prototype.

### Goal 3: Explain what the model sees

An AI poverty estimate is only useful if decision-makers can understand and trust it. A model that says "this village is poor" with no explanation is hard to act on. A model that says "this village scores low because it has low nighttime light levels, sparse rooftop density, and no visible road connections" is much more useful.

We apply **GradCAM** (Gradient-weighted Class Activation Mapping) to each satellite image after prediction. This produces a heatmap showing which pixels most influenced the model's output. If the bright regions on the heatmap land on rooftops, lit streets, and road networks, the model is picking up on meaningful poverty signals. If they land on empty fields or sky, something is wrong with the model.

We use this as both a quality check and a communication tool for policymakers.

---

## Data Sources

All satellite data is pulled through [Google Earth Engine](https://earthengine.google.com), a free platform for processing satellite images at global scale.

| Source | What it provides | Resolution | Years |
|---|---|---|---|
| VIIRS (NASA) | Nighttime lights | 500 m | 2014 to 2024 |
| Sentinel-2 (ESA) | Colour photos | 10 m | 2019 to present |
| Landsat 8/9 (NASA) | Colour photos | 30 m | 2014 to 2024 |
| Sentinel-1 (ESA) | Radar images | 10 m | 2014 to present |
| ESA WorldCover | Land use map | 10 m | 2021 |
| Meta HRSL | Population density | 30 m | 2019 |

Survey ground truth comes from DHS household surveys for **Kenya** (1,686 clusters, 2022 round) and **Nigeria** (1,362 clusters, 2023 round), giving a total of 3,048 labelled communities.

---

## Methodology

### Training in two stages

Training a deep learning model directly on 3,048 labelled survey points is too little data for a ResNet to generalise well. We follow a two-stage approach:

**Stage 1: Nighttime lights pretraining**
We first train the model to predict VIIRS nighttime light intensity from daytime Sentinel-2 photos. This task has millions of free training examples (the whole globe, every year). The model learns to recognise features linked to wealth and electrification (rooftops, road grids, lit buildings) without needing any survey labels.

**Stage 2: DHS fine-tuning**
We load the pretrained model and continue training it using the DHS wealth index as the target. The model already understands satellite imagery, so it can focus on learning the mapping from visual patterns to wealth scores with fewer examples.

### Evaluation

We evaluate models on a held-out test set that the model has never seen during training.

For the strictest evaluation, we use **leave-one-country-out cross-validation**: each test round trains on all countries except one, then tests on that held-out country. This tests whether the model has learned patterns that transfer to new countries it has never encountered, which is the scenario that actually matters for real-world deployment.

### Uncertainty

At inference time, we run the model 50 times with random dropout (some neurons switched off each run). The spread across those 50 predictions gives a confidence estimate. High spread means the model is uncertain about that community. Low spread means it is confident. This helps flag which clusters would most benefit from a real survey visit.

---

## Current Results

| Model | R² (held-out test set) |
|---|---|
| Gradient Boosting on 9 satellite features | **0.776** |
| ResNet18 CNN (existing checkpoints) | Negative (below baseline) |

The tabular Gradient Boosting model currently outperforms the CNN. This is expected: the CNN needs the nighttime-lights pretraining step to be completed before it can learn effectively from 3,048 training examples. The pretraining infrastructure is built and ready to run.

### Comparison to published research

| Study | Method | R² |
|---|---|---|
| Jean et al. 2016 | CNN with NTL pretraining | ~0.63 |
| Yeh et al. 2020 | Multi-task CNN | ~0.70 |
| **This project** | Gradient Boosting (tabular) | **0.776** |

Note: direct comparison is not straightforward because the datasets and countries differ. Our CNN, once properly pretrained, is the architecture most comparable to Jean et al. and Yeh et al.

---

## What This Project Contributes

1. A full end-to-end pipeline from raw satellite data to an interactive poverty map, built entirely on free and open data sources.

2. A deep learning training framework that supports ResNet18, Vision Transformer, multi-task, and time-series architectures on the same satellite patch dataset.

3. A 6-way satellite comparison experiment isolating the contribution of each sensor type to prediction accuracy.

4. A GradCAM explainability layer that produces spatial heatmaps for each prediction, supporting transparency and policy trust.

5. An interactive web map showing cluster-level poverty predictions, SDG progress scores, nighttime light trends, and conflict data for 28 African countries.

6. Honest documentation of current model limitations, including the gap between CNN and tabular performance, and a clear path to closing it.

---

## Limitations

**Real survey data is only available for Kenya and Nigeria.** The other 26 countries on the map use national-level satellite averages as inputs to the tabular model. Those predictions are rough estimates and should not be treated as survey-equivalent.

**No Pléiades (2 m per pixel) data.** This satellite requires a commercial licence. The project covers the Sentinel-2 (10 m) vs Landsat (30 m) resolution comparison. Pléiades access through ESA Third Party Missions is the planned next step.

**The CNN does not yet outperform the tabular model.** Completing the nighttime-lights pretraining step is the immediate next task. Once done, the CNN pipeline is wired and ready to replace the tabular model in the live map.

**The time-series model is not yet trained end-to-end.** Paired survey data across two or more years exists for Kenya and Nigeria. Training is the next step.

**SDG scores use estimated thresholds.** The thresholds for converting satellite signals into SDG progress scores were derived from the data itself, not calibrated against official UN monitoring benchmarks. They are useful for relative comparison but should not be reported as official statistics.

---

## Next Steps

1. Complete nighttime-lights pretraining and retrain the CNN on DHS labels.
2. Run leave-one-country-out evaluation and publish the results table.
3. Train the time-series model on Kenya (2014, 2022) and Nigeria (2018, 2023) paired rounds.
4. Expand DHS coverage to Ethiopia, Ghana, and Tanzania as data access is arranged.
5. Apply for Pléiades data access through ESA Third Party Missions to close the resolution comparison gap.
6. Cross-validate SDG thresholds against DHS household electricity access responses.
