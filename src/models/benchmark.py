"""Core benchmarking primitive: one manual repeated-K-Fold loop per model that yields,
in a single pass (no redundant refitting):
  - per-(repeat,fold) metrics (rmse/mae/medae/r2) for paired statistical tests,
  - fit/predict timing,
  - a (n_repeats x n_samples) out-of-fold prediction matrix -- each repeat is a full
    5-fold partition, so every sample gets exactly one OOF prediction per repeat. The
    per-sample mean across repeats gives a clean point estimate for residual diagnostics;
    the per-sample std across repeats gives a direct "prediction stability across folds"
    measure (item 7 of the phase spec), which sklearn's cross_val_predict cannot produce
    for a *repeated* CV scheme (it only supports a single partition).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone

from .configs import N_REPEATS, N_SPLITS, RNG_SEED
from .metrics import compute_metrics
from .pipelines import build_model_pipeline
from sklearn.model_selection import RepeatedKFold


def run_repeated_cv(model_name: str, X, y) -> dict:
    y = np.asarray(y)
    n = len(y)
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG_SEED)

    fold_rows = []
    oof = np.full((N_REPEATS, n), np.nan)
    fit_times, predict_times = [], []

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        repeat_idx = i // N_SPLITS
        fold_idx = i % N_SPLITS

        pipe = build_model_pipeline(model_name)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        t1 = time.perf_counter()
        preds = pipe.predict(X_test)
        t2 = time.perf_counter()

        fit_times.append(t1 - t0)
        predict_times.append(t2 - t1)
        oof[repeat_idx, test_idx] = preds

        m = compute_metrics(y_test, preds)
        fold_rows.append({"repeat": repeat_idx, "fold": fold_idx, **m})

    return {
        "model": model_name,
        "fold_rows": fold_rows,
        "oof_predictions": oof,  # shape (N_REPEATS, n) -- one full OOF partition per repeat
        "fit_time_mean": float(np.mean(fit_times)), "fit_time_std": float(np.std(fit_times)),
        "predict_time_mean": float(np.mean(predict_times)), "predict_time_std": float(np.std(predict_times)),
    }


def summarize(result: dict) -> dict:
    rows = result["fold_rows"]
    rmse = np.array([r["rmse"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    medae = np.array([r["medae"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    return {
        "model": result["model"],
        "rmse_mean": float(rmse.mean()), "rmse_std": float(rmse.std()),
        "mae_mean": float(mae.mean()), "mae_std": float(mae.std()),
        "medae_mean": float(medae.mean()), "medae_std": float(medae.std()),
        "r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
        "fit_time_mean": result["fit_time_mean"], "predict_time_mean": result["predict_time_mean"],
    }
