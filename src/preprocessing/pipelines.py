"""Model-aware pipeline factory.

`build_pipeline(...)` is the single place that assembles
FeatureSetSelector -> [outlier handler] -> [per-column skew transform] -> [global scaler] -> model
as one sklearn Pipeline object. Every stage is a proper fit/transform Estimator, so
cross_val_score / GridSearchCV / RepeatedKFold all refit every stage independently per fold --
this is what makes the whole thing leakage-safe (see reports/leakage_validation_report.md).
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from . import config as cfg
from .feature_selector import FeatureSetSelector
from .scalers import make_scaler
from .transformers import IQRClipper, SelectiveColumnTransform, Winsorizer, make_transform


def build_pipeline(
    model,
    feature_set: str = cfg.DEFAULT_FEATURE_SET,
    scaler: str = "none",
    skew_transform: str = "none",
    skew_columns: list[str] | None = None,
    outlier_strategy: str = "none",
):
    """Build a full leakage-safe sklearn Pipeline.

    Parameters
    ----------
    model : sklearn-compatible estimator (unfitted)
    feature_set : name from preprocessing.config.FEATURE_SETS
    scaler : name from preprocessing.scalers.make_scaler
    skew_transform : name applied to `skew_columns` only (rest pass through unchanged)
    skew_columns : which columns get `skew_transform`; defaults to config.SKEW_CANDIDATE_COLUMNS
    outlier_strategy : "none" | "winsorize_1_99" | "clip_iqr"
    """
    feature_names = cfg.FEATURE_SETS[feature_set]
    steps = [("select_features", FeatureSetSelector(feature_names))]

    if outlier_strategy == "winsorize_1_99":
        steps.append(("outliers", Winsorizer(1.0, 99.0)))
    elif outlier_strategy == "clip_iqr":
        steps.append(("outliers", IQRClipper(1.5)))
    elif outlier_strategy != "none":
        raise ValueError(f"Unknown outlier_strategy: {outlier_strategy}")

    if skew_transform != "none":
        cols = skew_columns if skew_columns is not None else cfg.SKEW_CANDIDATE_COLUMNS
        cols = [c for c in cols if c in feature_names]
        if cols:
            steps.append(("skew_transform", SelectiveColumnTransform(cols, skew_transform)))

    scaler_obj = make_scaler(scaler)
    if scaler_obj is not None:
        steps.append(("scale", scaler_obj))

    steps.append(("model", model))
    return Pipeline(steps)
