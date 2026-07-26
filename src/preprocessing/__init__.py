"""Leakage-safe, model-aware preprocessing for the reactor yield hackathon project.

Everything importable from here is designed to live inside a scikit-learn Pipeline so
that cross-validation refits preprocessing independently on every fold. See
reports/preprocessing_report.md, reports/leakage_validation_report.md, and
reports/pipeline_benchmark_report.md for the evidence behind every default in config.py.
"""
from . import config
from .feature_selector import FeatureSetSelector
from .pipelines import build_pipeline
from .scalers import make_scaler
from .transformers import IQRClipper, SelectiveColumnTransform, Winsorizer, make_transform

__all__ = [
    "config", "FeatureSetSelector", "build_pipeline", "make_scaler",
    "make_transform", "SelectiveColumnTransform", "Winsorizer", "IQRClipper",
]
