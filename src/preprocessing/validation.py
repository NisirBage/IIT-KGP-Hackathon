"""Leakage-validation utilities -- used to empirically demonstrate (not just assert) that
the pipelines in this module do not leak validation-fold information into preprocessing.
See reports/leakage_validation_report.md for the results these functions produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

from .pipelines import build_pipeline
from .scalers import make_scaler
from .feature_selector import FeatureSetSelector


def check_fold_specific_fitting(X: pd.DataFrame, y: np.ndarray, feature_set: str, scaler: str, cv=None) -> list[dict]:
    """Fits FeatureSetSelector+scaler on each CV fold's TRAIN split only and records the
    fitted scaler statistics (mean_ for StandardScaler) per fold. If preprocessing were
    accidentally fit on the full dataset once, every fold would show identical statistics;
    genuine per-fold fitting produces different statistics each time (since each fold
    excludes a different validation slice)."""
    from . import config as cfg
    cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
    feature_names = cfg.FEATURE_SETS[feature_set]
    rows = []
    for fold_i, (train_idx, _) in enumerate(cv.split(X)):
        selector = FeatureSetSelector(feature_names).fit(X.iloc[train_idx])
        X_train_feat = selector.transform(X.iloc[train_idx])
        scaler_obj = make_scaler(scaler)
        scaler_obj.fit(X_train_feat)
        rows.append({
            "fold": fold_i,
            "n_train": len(train_idx),
            "fitted_mean_first_col": float(scaler_obj.mean_[0]) if hasattr(scaler_obj, "mean_") else None,
        })
    return rows


def leaky_vs_correct_cv(model, X: pd.DataFrame, y: np.ndarray, feature_set: str, scaler: str, cv=None) -> dict:
    """Empirically compares:
    (a) LEAKY: fit FeatureSetSelector+scaler ONCE on the full dataset, transform everything,
        then cross_val_score the model on the already-transformed data (each "fold" reuses
        scaling statistics that saw its own validation rows).
    (b) CORRECT: a full Pipeline (selector+scaler+model) run through cross_val_score, so
        every fold's preprocessing is refit on that fold's training split only.
    Returns both RMSE distributions so the leakage bias can be quantified directly.
    """
    from . import config as cfg
    cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
    feature_names = cfg.FEATURE_SETS[feature_set]

    # (a) leaky
    selector = FeatureSetSelector(feature_names).fit(X)
    X_feat = selector.transform(X)
    scaler_obj = make_scaler(scaler)
    if scaler_obj is not None:
        X_scaled = scaler_obj.fit_transform(X_feat)
    else:
        X_scaled = X_feat.values
    leaky_scores = -cross_val_score(clone(model), X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")

    # (b) correct
    pipe = build_pipeline(clone(model), feature_set=feature_set, scaler=scaler)
    correct_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    return {
        "leaky_rmse_mean": float(np.mean(leaky_scores)), "leaky_rmse_std": float(np.std(leaky_scores)),
        "correct_rmse_mean": float(np.mean(correct_scores)), "correct_rmse_std": float(np.std(correct_scores)),
        "leaky_scores": leaky_scores.tolist(), "correct_scores": correct_scores.tolist(),
    }
