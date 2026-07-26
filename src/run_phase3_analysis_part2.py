"""
Phase 3 continuation: the previous run (run_phase3_analysis.py) was killed partway through
by the environment before finishing the outlier/leakage/main-benchmark sections (scaler
benchmark, tree-scaling confirmation, and distribution-transform results were already
captured from its log). This script picks up exactly where it left off and uses a lighter
CV budget (5x5 instead of 5x10) for the tree-model grids specifically to reduce runtime risk.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import FEATURE_COLS, TARGET_COL, load_train  # noqa: E402
from preprocessing import config as cfg  # noqa: E402
from preprocessing.pipelines import build_pipeline  # noqa: E402
from preprocessing.validation import check_fold_specific_fitting, leaky_vs_correct_cv  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "pipelines"
RNG_SEED = 42
np.random.seed(RNG_SEED)

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values

results: dict = {}
CV_LIGHT = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG_SEED)  # reduced from 5x10

# ===========================================================================
# 5. OUTLIER STRATEGY benchmark (full: Ridge/RandomForest/CatBoost x 3 strategies)
# ===========================================================================
outlier_rows = []
for model_name, model in {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RNG_SEED),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=RNG_SEED),
}.items():
    for strat in cfg.OUTLIER_STRATEGIES:
        scaler_name = "standard" if model_name == "Ridge" else "none"
        pipe = build_pipeline(model, feature_set="core", scaler=scaler_name, outlier_strategy=strat)
        scores = -cross_val_score(pipe, X, y, cv=CV_LIGHT, scoring="neg_root_mean_squared_error")
        outlier_rows.append({
            "model": model_name, "outlier_strategy": strat,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[outlier] {model_name:14s} {strat:16s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}", flush=True)
results["outlier_strategy_benchmark"] = outlier_rows

# ===========================================================================
# 7. LEAKAGE VALIDATION
# ===========================================================================
fold_check = check_fold_specific_fitting(X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=RNG_SEED))
results["leakage_fold_specific_fitting"] = fold_check
print("[leakage] fold-specific fitted means:", [round(r["fitted_mean_first_col"], 3) for r in fold_check], flush=True)

leak_cmp = leaky_vs_correct_cv(Ridge(alpha=1.0), X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=RNG_SEED))
results["leakage_leaky_vs_correct"] = leak_cmp
print(f"[leakage] leaky RMSE={leak_cmp['leaky_rmse_mean']:.4f}  correct RMSE={leak_cmp['correct_rmse_mean']:.4f}", flush=True)

leak_repeats = []
for seed in range(30):
    cmp = leaky_vs_correct_cv(Ridge(alpha=1.0), X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    leak_repeats.append({"seed": seed, "leaky_rmse": cmp["leaky_rmse_mean"], "correct_rmse": cmp["correct_rmse_mean"]})
results["leakage_repeated_comparison"] = leak_repeats
diffs = [r["leaky_rmse"] - r["correct_rmse"] for r in leak_repeats]
print(f"[leakage] mean(leaky-correct) over 30 seeds = {np.mean(diffs):.4f} (negative = leaky is optimistically biased)", flush=True)

with open(REPORTS / "phase3_analysis_results_part2a.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("PART2A (outlier+leakage) complete.", flush=True)
