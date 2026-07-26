"""
Phase 3: Preprocessing Pipeline, Leakage Prevention & Model Readiness.

Runs: (1) a distributional audit of the recommended feature set, (2) a scaler benchmark
for scale-sensitive model families, (3) a distribution-transform benchmark for the 3
flagged skewed columns, (4) an outlier-strategy benchmark, (5) an explicit leakage
validation (fold-specific fitting + leaky-vs-correct CV comparison), (6) the main
preprocessing-combination benchmark (raw/scaled/transformed/scaled+transformed) across
Ridge/RandomForest/CatBoost, and (7) a pipeline serialization round-trip test.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import FEATURE_COLS, TARGET_COL, load_train  # noqa: E402
from features import build_feature_frame, RECOMMENDED_FEATURES_CORE  # noqa: E402
from preprocessing import config as cfg  # noqa: E402
from preprocessing.pipelines import build_pipeline  # noqa: E402
from preprocessing.transformers import columns_valid_for_transform  # noqa: E402
from preprocessing.validation import check_fold_specific_fitting, leaky_vs_correct_cv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "pipelines"
RNG_SEED = 42
np.random.seed(RNG_SEED)

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values
n = len(y)

core_engineered = build_feature_frame(train, RECOMMENDED_FEATURES_CORE)
core_full = pd.concat([train[FEATURE_COLS], core_engineered], axis=1)

results: dict = {}
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)

# ===========================================================================
# 2. AUDIT the recommended feature set
# ===========================================================================
audit_rows = []
for col in core_full.columns:
    x = core_full[col].values
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((x < lo) | (x > hi)).sum())
    audit_rows.append({
        "feature": col,
        "n_missing": int(pd.isna(x).sum()),
        "skewness": float(stats.skew(x)),
        "kurtosis_excess": float(stats.kurtosis(x)),  # excess kurtosis (0 = normal)
        "min": float(x.min()), "max": float(x.max()),
        "std": float(x.std(ddof=1)), "variance": float(x.var(ddof=1)),
        "n_outliers_iqr": n_out,
        "strictly_positive": bool(np.all(x > 0)),
        "has_zero_or_negative": bool(np.any(x <= 0)),
    })
results["feature_audit"] = audit_rows

# ===========================================================================
# 3. SCALING STRATEGY benchmark -- scale-sensitive models only, core feature set
# ===========================================================================
scale_sensitive_models = {
    "Ridge": Ridge(alpha=1.0),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=10),
    "GaussianProcess": GaussianProcessRegressor(
        kernel=RBF() + WhiteKernel(), random_state=RNG_SEED, normalize_y=True,
    ),
}

scaler_rows = []
for model_name, model in scale_sensitive_models.items():
    for scaler_name in cfg.SCALER_NAMES:
        if scaler_name == "power_boxcox":
            # core feature set includes signed columns (delta_T, severity_index) --
            # Box-Cox requires strictly positive input across ALL columns fed to it.
            continue
        pipe = build_pipeline(model, feature_set="core", scaler=scaler_name)
        scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
        scaler_rows.append({
            "model": model_name, "scaler": scaler_name,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[scaler] {model_name:16s} {scaler_name:18s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}")

# Box-Cox separately on raw_only (all raw features are strictly positive)
for model_name, model in scale_sensitive_models.items():
    pipe = build_pipeline(model, feature_set="raw_only", scaler="power_boxcox")
    scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
    scaler_rows.append({
        "model": model_name, "scaler": "power_boxcox (raw_only feature set)",
        "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
    })
    print(f"[scaler,raw_only] {model_name:16s} power_boxcox       RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}")

results["scaling_benchmark"] = scaler_rows

# Tree models: confirmatory check that scaling doesn't matter (expected, quick)
from catboost import CatBoostRegressor
tree_models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RNG_SEED),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=RNG_SEED),
}
tree_scaler_rows = []
for model_name, model in tree_models.items():
    for scaler_name in ["none", "standard", "robust"]:
        pipe = build_pipeline(model, feature_set="core", scaler=scaler_name)
        scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
        tree_scaler_rows.append({
            "model": model_name, "scaler": scaler_name,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[tree-scaler-check] {model_name:16s} {scaler_name:10s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}")
results["tree_scaling_confirmation"] = tree_scaler_rows

# ===========================================================================
# 4. DISTRIBUTION TRANSFORMS on the 3 flagged skewed columns
# ===========================================================================
transform_rows = []
for col in cfg.SKEW_CANDIDATE_COLUMNS:
    x = core_full[col].values
    base_skew = float(stats.skew(x))
    for tname in cfg.TRANSFORM_NAMES:
        if tname == "none":
            transform_rows.append({
                "column": col, "transform": "none", "skew_before": base_skew,
                "skew_after": base_skew, "valid": True,
                "ridge_rmse_mean": None, "ridge_rmse_std": None,
            })
            continue
        valid_cols = columns_valid_for_transform(core_full, [col], tname)
        if not valid_cols:
            transform_rows.append({
                "column": col, "transform": tname, "skew_before": base_skew,
                "skew_after": None, "valid": False,
                "ridge_rmse_mean": None, "ridge_rmse_std": None,
                "reason_invalid": "column contains values outside the transform's valid domain (e.g. <=0)",
            })
            continue
        from preprocessing.transformers import make_transform
        t = make_transform(tname)
        x_t = t.fit_transform(x.reshape(-1, 1)).ravel()
        skew_after = float(stats.skew(x_t))

        pipe = build_pipeline(
            Ridge(alpha=1.0), feature_set="core", scaler="standard",
            skew_transform=tname, skew_columns=[col],
        )
        scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
        transform_rows.append({
            "column": col, "transform": tname, "skew_before": base_skew,
            "skew_after": skew_after, "valid": True,
            "ridge_rmse_mean": float(np.mean(scores)), "ridge_rmse_std": float(np.std(scores)),
        })
        print(f"[transform] {col:16s} {tname:14s} skew {base_skew:+.3f}->{skew_after:+.3f} Ridge RMSE={np.mean(scores):7.3f}")

# baseline: core set, standard-scaled, no skew transform at all
pipe_base = build_pipeline(Ridge(alpha=1.0), feature_set="core", scaler="standard", skew_transform="none")
base_scores = -cross_val_score(pipe_base, X, y, cv=CV, scoring="neg_root_mean_squared_error")
results["transform_baseline_ridge_rmse"] = {"mean": float(np.mean(base_scores)), "std": float(np.std(base_scores))}
results["distribution_transforms"] = transform_rows

# ===========================================================================
# 5. OUTLIER STRATEGY benchmark
# ===========================================================================
outlier_rows = []
for model_name, model in {"Ridge": Ridge(alpha=1.0), "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RNG_SEED), "CatBoost": CatBoostRegressor(verbose=False, random_state=RNG_SEED)}.items():
    for strat in cfg.OUTLIER_STRATEGIES:
        scaler_name = "standard" if model_name == "Ridge" else "none"
        pipe = build_pipeline(model, feature_set="core", scaler=scaler_name, outlier_strategy=strat)
        scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
        outlier_rows.append({
            "model": model_name, "outlier_strategy": strat,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[outlier] {model_name:14s} {strat:16s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}")
results["outlier_strategy_benchmark"] = outlier_rows

# ===========================================================================
# 7. LEAKAGE VALIDATION
# ===========================================================================
fold_check = check_fold_specific_fitting(X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=RNG_SEED))
results["leakage_fold_specific_fitting"] = fold_check

leak_cmp = leaky_vs_correct_cv(Ridge(alpha=1.0), X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=RNG_SEED))
results["leakage_leaky_vs_correct"] = leak_cmp
print(f"[leakage] leaky RMSE={leak_cmp['leaky_rmse_mean']:.4f}  correct RMSE={leak_cmp['correct_rmse_mean']:.4f}")

# Repeat the leaky-vs-correct comparison many times (different random KFold splits) to see
# whether the leaky estimate is *systematically* optimistic or just noise from one split.
leak_repeats = []
for seed in range(30):
    cmp = leaky_vs_correct_cv(Ridge(alpha=1.0), X, y, feature_set="core", scaler="standard", cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    leak_repeats.append({"seed": seed, "leaky_rmse": cmp["leaky_rmse_mean"], "correct_rmse": cmp["correct_rmse_mean"]})
results["leakage_repeated_comparison"] = leak_repeats
diffs = [r["leaky_rmse"] - r["correct_rmse"] for r in leak_repeats]
print(f"[leakage] mean(leaky-correct) over 30 seeds = {np.mean(diffs):.4f} (negative = leaky is optimistically biased)")

# ===========================================================================
# 6/8. MAIN PIPELINE BENCHMARK: raw / scaled / transformed / scaled+transformed
# ===========================================================================
main_bench_rows = []
configs = {
    "raw": dict(scaler="none", skew_transform="none"),
    "scaled": dict(scaler="standard", skew_transform="none"),
    "transformed": dict(scaler="none", skew_transform="yeo_johnson"),
    "scaled_plus_transformed": dict(scaler="standard", skew_transform="yeo_johnson"),
}
main_models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RNG_SEED),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=RNG_SEED),
}
for cfg_name, cfg_kwargs in configs.items():
    for model_name, model in main_models.items():
        pipe = build_pipeline(model, feature_set="core", **cfg_kwargs)
        scores = -cross_val_score(pipe, X, y, cv=CV, scoring="neg_root_mean_squared_error")
        main_bench_rows.append({
            "preprocessing_config": cfg_name, "model": model_name,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[main-bench] {cfg_name:24s} {model_name:14s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}")
results["main_pipeline_benchmark"] = main_bench_rows

# ===========================================================================
# 9. PIPELINE SERIALIZATION test
# ===========================================================================
import joblib

final_pipe = build_pipeline(Ridge(alpha=1.0), feature_set="core", scaler="standard")
final_pipe.fit(X, y)
preds_before = final_pipe.predict(X)

ARTIFACTS.mkdir(parents=True, exist_ok=True)
pipeline_path = ARTIFACTS / "ridge_core_standard_v1.joblib"
joblib.dump(final_pipe, pipeline_path)
loaded_pipe = joblib.load(pipeline_path)
preds_after = loaded_pipe.predict(X)

serialization_ok = bool(np.allclose(preds_before, preds_after))
results["serialization_test"] = {
    "path": str(pipeline_path), "predictions_match": serialization_ok,
    "max_abs_diff": float(np.max(np.abs(preds_before - preds_after))),
}
print(f"[serialization] predictions match after reload: {serialization_ok}")

# ===========================================================================
# Dump
# ===========================================================================
with open(REPORTS / "phase3_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nPhase 3 analysis complete.")
print(f"Results JSON: {REPORTS / 'phase3_analysis_results.json'}")
