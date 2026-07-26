"""Statistical model comparison -- RMSE ranking alone is not treated as sufficient.

All pairwise tests assume the per-fold RMSE vectors are PAIRED across models (same fold,
same held-out rows) -- true here because every model in the registry is run through the
identical RepeatedKFold(seed=42) sequence in benchmark.run_repeated_cv.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def fold_rmse_vector(fold_rows: list[dict]) -> np.ndarray:
    """Per-fold RMSE in (repeat, fold) order -- the paired unit for all tests below."""
    rows = sorted(fold_rows, key=lambda r: (r["repeat"], r["fold"]))
    return np.array([r["rmse"] for r in rows])


def paired_comparison(rmse_a: np.ndarray, rmse_b: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon signed-rank + bootstrap CI on (rmse_a - rmse_b)."""
    diff = rmse_a - rmse_b
    t_stat, t_p = stats.ttest_rel(rmse_a, rmse_b)
    try:
        w_stat, w_p = stats.wilcoxon(rmse_a, rmse_b)
    except ValueError:
        w_stat, w_p = np.nan, 1.0  # identical arrays (zero differences) -- Wilcoxon undefined

    rng = np.random.default_rng(42)
    n = len(diff)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(5000)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_diff": float(diff.mean()), "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
        "bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    }


def friedman_test(rmse_matrix: np.ndarray, model_names: list[str]) -> dict:
    """rmse_matrix: shape (n_folds, n_models), same fold order for every column."""
    stat, p = stats.friedmanchisquare(*[rmse_matrix[:, i] for i in range(rmse_matrix.shape[1])])
    return {"friedman_stat": float(stat), "friedman_p": float(p), "models": model_names}


def nemenyi_posthoc(rmse_matrix: np.ndarray, model_names: list[str]):
    """Nemenyi post-hoc test (only meaningful if the Friedman test is significant).
    Returns a DataFrame of pairwise p-values, or None if scikit-posthocs is unavailable."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        return None
    result = sp.posthoc_nemenyi_friedman(rmse_matrix)
    result.columns = model_names
    result.index = model_names
    return result
