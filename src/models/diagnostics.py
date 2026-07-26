"""Residual diagnostics, prediction-behavior checks, and feature importance -- run on the
single-repeat-averaged out-of-fold predictions produced by benchmark.run_repeated_cv.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.inspection import permutation_importance


def oof_point_estimates(oof: np.ndarray) -> dict:
    """oof: shape (n_repeats, n_samples), NaN where a sample wasn't in that repeat's fold
    split (never happens here since every repeat is a full partition, but guarded anyway).
    Returns per-sample mean (point estimate) and std (stability across repeats)."""
    mean = np.nanmean(oof, axis=0)
    std = np.nanstd(oof, axis=0)
    return {"mean": mean, "std": std}


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    shapiro_stat, shapiro_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    bp_corr, bp_p = stats.spearmanr(np.abs(resid), y_pred)  # heteroscedasticity proxy: |resid| vs prediction
    return {
        "mean_residual": float(resid.mean()),
        "std_residual": float(resid.std(ddof=1)),
        "skew_residual": float(stats.skew(resid)),
        "kurtosis_residual_excess": float(stats.kurtosis(resid)),
        "shapiro_stat": float(shapiro_stat), "shapiro_p": float(shapiro_p),
        "heteroscedasticity_spearman_abs_resid_vs_pred": float(bp_corr),
        "heteroscedasticity_p": float(bp_p),
    }


def prediction_behavior(y_true: np.ndarray, oof_mean: np.ndarray, oof_std: np.ndarray) -> dict:
    """Physical-plausibility + stability checks. Target is a percentage yield in [0, 100]."""
    n_below_0 = int(np.sum(oof_mean < 0))
    n_above_100 = int(np.sum(oof_mean > 100))
    return {
        "pred_min": float(oof_mean.min()), "pred_max": float(oof_mean.max()),
        "n_predictions_below_0": n_below_0, "n_predictions_above_100": n_above_100,
        "pct_physically_implausible": float((n_below_0 + n_above_100) / len(oof_mean) * 100),
        "mean_fold_to_fold_std": float(oof_std.mean()),  # average per-sample std across the 10 repeats
        "max_fold_to_fold_std": float(oof_std.max()),
    }


def compute_importances(pipeline, X, y, feature_names: list[str], raw_feature_names: list[str] | None = None) -> dict:
    """Impurity importance (tree models only, keyed by the pipeline's ENGINEERED feature
    names) + permutation importance (any model, keyed by RAW column names -- sklearn's
    permutation_importance permutes columns of whatever X is passed to the full pipeline,
    which is the raw DataFrame here, since feature engineering happens inside the pipeline).
    """
    out = {}
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        out["impurity_importance"] = dict(zip(feature_names, model.feature_importances_.tolist()))

    raw_names = raw_feature_names if raw_feature_names is not None else list(X.columns)
    perm = permutation_importance(pipeline, X, y, n_repeats=20, random_state=42, scoring="neg_root_mean_squared_error")
    out["permutation_importance_mean"] = dict(zip(raw_names, perm.importances_mean.tolist()))
    out["permutation_importance_std"] = dict(zip(raw_names, perm.importances_std.tolist()))
    return out
