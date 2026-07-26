"""Builds each registered model's full leakage-safe pipeline using its registry-assigned
preprocessing settings -- the single place Phase 4 turns a registry entry into a fittable
sklearn Pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.pipelines import build_pipeline  # noqa: E402

from .registry import MODEL_REGISTRY


def build_model_pipeline(model_name: str):
    entry = MODEL_REGISTRY[model_name]
    model = entry["factory"]()
    return build_pipeline(model, feature_set=entry["feature_set"], scaler=entry["scaler"])
