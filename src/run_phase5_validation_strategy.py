"""
Phase 5, Step 1: Validation Strategy Comparison.

Compares RepeatedKFold (Phase 3/4 baseline), Nested CV, Monte Carlo CV (repeated random
splits), and Bootstrap (out-of-bag) validation on ExtraTrees (Phase 4's top model, fixed at
its Phase-4 default hyperparameters throughout -- this step is about the PROTOCOL, not the
model). Measures variance of the RMSE estimate, computational cost, and susceptibility to
optimistic bias (via a direct naive-vs-nested comparison, mirroring the Phase 3 leakage
demonstration).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models.pipelines import build_model_pipeline  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RepeatedKFold, KFold, ShuffleSplit, cross_val_score, GridSearchCV,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
RNG_SEED = 42

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values
n = len(y)

results = {}

def rmse_scores(pipe, X, y, cv):
    return -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")

# ===========================================================================
# A. RepeatedKFold(5,10) -- baseline, reuse Phase 4 protocol exactly
# ===========================================================================
t0 = time.perf_counter()
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)
scores_rkf = rmse_scores(build_model_pipeline("ExtraTrees"), X, y, cv)
t_rkf = time.perf_counter() - t0
results["repeated_kfold_5x10"] = {
    "n_fits": 50, "wall_time_sec": t_rkf,
    "rmse_mean": float(scores_rkf.mean()), "rmse_std": float(scores_rkf.std()),
}
print(f"RepeatedKFold(5,10): RMSE={scores_rkf.mean():.3f}+/-{scores_rkf.std():.3f}  n_fits=50  time={t_rkf:.1f}s")

# Stability of the *summary statistic itself*: repeat the whole 5x10 protocol under 20
# different base seeds and look at how much the resulting mean RMSE wobbles -- this is the
# "would we get a different answer if we'd gotten unlucky with the seed" question.
reseed_means = []
for seed in range(20):
    cv_s = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
    s = rmse_scores(build_model_pipeline("ExtraTrees"), X, y, cv_s)
    reseed_means.append(float(s.mean()))
results["repeated_kfold_5x10"]["reseed_mean_std"] = float(np.std(reseed_means))
results["repeated_kfold_5x10"]["reseed_means"] = reseed_means
print(f"  reseed (20x) mean-RMSE std = {np.std(reseed_means):.4f}  range=[{min(reseed_means):.2f},{max(reseed_means):.2f}]")

# ===========================================================================
# B. Monte Carlo CV (ShuffleSplit) -- 50 random 80/20 splits, same total fit count as A
# ===========================================================================
t0 = time.perf_counter()
cv_mc = ShuffleSplit(n_splits=50, test_size=0.2, random_state=RNG_SEED)
scores_mc = rmse_scores(build_model_pipeline("ExtraTrees"), X, y, cv_mc)
t_mc = time.perf_counter() - t0
results["monte_carlo_50x"] = {
    "n_fits": 50, "wall_time_sec": t_mc,
    "rmse_mean": float(scores_mc.mean()), "rmse_std": float(scores_mc.std()),
}
print(f"MonteCarloCV(50x, 80/20): RMSE={scores_mc.mean():.3f}+/-{scores_mc.std():.3f}  n_fits=50  time={t_mc:.1f}s")

# ===========================================================================
# C. Bootstrap (out-of-bag) validation -- 50 bootstrap resamples, evaluate on OOB rows
# ===========================================================================
t0 = time.perf_counter()
rng = np.random.default_rng(RNG_SEED)
boot_rmses = []
for b in range(50):
    idx = rng.integers(0, n, size=n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[np.unique(idx)] = False
    if oob_mask.sum() < 10:
        continue
    pipe = build_model_pipeline("ExtraTrees")
    pipe.fit(X.iloc[idx], y[idx])
    pred = pipe.predict(X.iloc[oob_mask])
    rmse = float(np.sqrt(np.mean((y[oob_mask] - pred) ** 2)))
    boot_rmses.append(rmse)
t_boot = time.perf_counter() - t0
results["bootstrap_oob_50x"] = {
    "n_fits": len(boot_rmses), "wall_time_sec": t_boot,
    "rmse_mean": float(np.mean(boot_rmses)), "rmse_std": float(np.std(boot_rmses)),
    "mean_oob_fraction": float(np.mean([1 for _ in boot_rmses]) if boot_rmses else 0),
}
print(f"Bootstrap-OOB(50x): RMSE={np.mean(boot_rmses):.3f}+/-{np.std(boot_rmses):.3f}  n_fits={len(boot_rmses)}  time={t_boot:.1f}s")

# ===========================================================================
# D. Nested CV vs. naive (non-nested) tune-then-report -- quantifies optimistic bias
# ===========================================================================
# A small, cheap grid (this is about the PROTOCOL's bias, not finding the best model)
param_grid = {"model__max_depth": [5, 10, 20, None], "model__min_samples_leaf": [1, 3, 5]}

t0 = time.perf_counter()
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RNG_SEED)
nested_scores = []
naive_inner_best_scores = []
for train_idx, test_idx in outer_cv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    gs = GridSearchCV(build_model_pipeline("ExtraTrees"), param_grid, cv=inner_cv,
                       scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    naive_inner_best_scores.append(-gs.best_score_)  # the biased "in-sample-search" estimate
    pred = gs.predict(X_te)
    nested_scores.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))  # the honest outer estimate
t_nested = time.perf_counter() - t0

results["nested_cv_5x3"] = {
    "n_fits_approx": 5 * 3 * (4 * 3),  # outer folds * inner folds * grid size
    "wall_time_sec": t_nested,
    "outer_unbiased_rmse_mean": float(np.mean(nested_scores)), "outer_unbiased_rmse_std": float(np.std(nested_scores)),
    "naive_inner_selected_rmse_mean": float(np.mean(naive_inner_best_scores)),
    "optimistic_bias": float(np.mean(naive_inner_best_scores) - np.mean(nested_scores)),
}
print(f"Nested CV(5x3): outer(honest) RMSE={np.mean(nested_scores):.3f}+/-{np.std(nested_scores):.3f}")
print(f"  naive inner-selected RMSE={np.mean(naive_inner_best_scores):.3f}  optimistic bias={np.mean(naive_inner_best_scores)-np.mean(nested_scores):+.3f}")
print(f"  time={t_nested:.1f}s")

with open(REPORTS / "phase5_validation_strategy_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nValidation strategy comparison complete.")
