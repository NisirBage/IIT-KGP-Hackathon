"""Phase 3 continuation, part 3: main preprocessing-combination benchmark + serialization test."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from preprocessing.pipelines import build_pipeline  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
import joblib  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "pipelines"
RNG_SEED = 42
np.random.seed(RNG_SEED)

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values

results: dict = {}
CV_LIGHT = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RNG_SEED)

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
        scores = -cross_val_score(pipe, X, y, cv=CV_LIGHT, scoring="neg_root_mean_squared_error")
        main_bench_rows.append({
            "preprocessing_config": cfg_name, "model": model_name,
            "rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores)),
        })
        print(f"[main-bench] {cfg_name:24s} {model_name:14s} RMSE={np.mean(scores):7.3f} +/- {np.std(scores):.3f}", flush=True)
results["main_pipeline_benchmark"] = main_bench_rows

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
print(f"[serialization] predictions match after reload: {serialization_ok}", flush=True)

with open(REPORTS / "phase3_analysis_results_part3.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("PART3 (main benchmark + serialization) complete.", flush=True)
