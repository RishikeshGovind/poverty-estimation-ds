"""
SDG Scoring Layer.

Maps raw model outputs (wealth index, nighttime lights, S2 brightness) to
0–100 SDG progress scores using defensible linear thresholds.

Scoring logic
-------------
SDG 1 — No Poverty
  Input : DHS wealth index (≈ −2 poorest → +2 wealthiest)
  Method: linear map over [config.scoring.sdg1.min, config.scoring.sdg1.max]

SDG 7 — Clean Energy (proxy: nighttime lights)
  Input : VIIRS NTL normalised to [0, 1]
  Method: clip(ntl / ntl_threshold, 0, 1) × 100
  Rationale: values above the threshold indicate reliable electricity access

SDG 11 — Sustainable Cities (proxy: S2 mean brightness)
  Input : S2 mean brightness normalised to [0, 1]
  Method: clip(brightness / buildup_threshold, 0, 1) × 100
  Rationale: higher brightness correlates with built-up area / infrastructure

Composite
  Weighted mean of available task scores using task weights from config.
"""

import numpy as np
from utils.config import load_config


class SDGScorer:
    def __init__(self):
        cfg = load_config()
        sc = cfg["scoring"]
        self.sdg1_min = sc["sdg1"]["min"]
        self.sdg1_max = sc["sdg1"]["max"]
        self.sdg7_thr = sc["sdg7"]["ntl_threshold"]
        self.sdg11_thr = sc["sdg11"]["buildup_threshold"]
        self.task_weights = cfg["tasks"]["weights"]

    def sdg1(self, wealth_index) -> float:
        """DHS wealth index → SDG 1 score [0, 100]."""
        val = np.asarray(wealth_index, dtype=float)
        return float(np.clip((val - self.sdg1_min) / (self.sdg1_max - self.sdg1_min), 0, 1) * 100)

    def sdg7(self, ntl_normalised) -> float:
        """VIIRS NTL (0–1) → SDG 7 score [0, 100]."""
        val = np.asarray(ntl_normalised, dtype=float)
        if np.isnan(val):
            return float("nan")
        return float(np.clip(val / self.sdg7_thr, 0, 1) * 100)

    def sdg11(self, buildup_normalised) -> float:
        """S2 brightness (0–1) → SDG 11 score [0, 100]."""
        val = np.asarray(buildup_normalised, dtype=float)
        if np.isnan(val):
            return float("nan")
        return float(np.clip(val / self.sdg11_thr, 0, 1) * 100)

    def composite(self, sdg1_score, sdg7_score=None, sdg11_score=None) -> float:
        """Weighted average of available scores."""
        scores, weights = [sdg1_score], [self.task_weights.get("sdg1_wealth", 1.0)]
        if sdg7_score is not None and not np.isnan(sdg7_score):
            scores.append(sdg7_score)
            weights.append(self.task_weights.get("sdg7_ntl", 0.5))
        if sdg11_score is not None and not np.isnan(sdg11_score):
            scores.append(sdg11_score)
            weights.append(self.task_weights.get("sdg11_buildup", 0.5))
        total_w = sum(weights)
        return float(sum(s * w for s, w in zip(scores, weights)) / total_w)

    def score_row(self, row: dict) -> dict:
        """Score a single prediction row dict. Returns dict of scores."""
        s1 = self.sdg1(row.get("prediction", row.get("sdg1_wealth", row.get("label", 0))))
        s7 = self.sdg7(row["sdg7_ntl"]) if "sdg7_ntl" in row and not _isnan(row["sdg7_ntl"]) else None
        s11 = self.sdg11(row["sdg11_buildup"]) if "sdg11_buildup" in row and not _isnan(row["sdg11_buildup"]) else None
        return {
            "sdg1_score":  round(s1, 1),
            "sdg7_score":  round(s7, 1) if s7 is not None else None,
            "sdg11_score": round(s11, 1) if s11 is not None else None,
            "composite_score": round(self.composite(s1, s7, s11), 1),
        }


def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True
