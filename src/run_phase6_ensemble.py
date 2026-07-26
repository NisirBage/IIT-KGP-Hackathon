"""
Phase 6: Evidence-based ensemble go/no-go evaluation.

Step 1: build fresh per-repeat OOF prediction matrices (RepeatedKFold(5,10), identical
protocol/seed to Phase 4/5) for the 4 SELECTED configurations:
  - ExtraTrees: Phase 4 defaults (Phase 5 proved tuning hurt this model -- reuse the
    existing Phase 4 checkpoint directly rather than refitting)
  - CatBoost, RandomForest, GaussianProcess: Phase 5 tuned best params (Phase 5 proved
    tuning helped these three)

Step 2: error diversity beyond prediction correlation (residual correlation, disagreement
rate, operating-region disagreement, zero-yield disagreement, thermal-threshold-region
disagreement).

Step 3: weighted averaging (equal / RMSE-weighted / inverse-variance-weighted) evaluated
directly on the fresh OOF matrices -- fold-level RMSE, no additional fitting needed since
weights are deterministic functions of already-known, already-validated performance, not
fit on this data.

Step 4/5: a linear blender and one stacking architecture (ExtraTrees+CatBoost+RandomForest
-> Ridge), BOTH trained on repeat-0's OOF predictions only and evaluated on repeats 1-9
(45 folds never used to fit the meta-model) -- a leakage-free, cheap alternative to full
nested CV, documented explicitly in ensemble_evaluation_report.md.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models.registry import MODEL_REGISTRY  # noqa: E402
from models.metrics import compute_metrics  # noqa: E402
from preprocessing.pipelines import build_pipeline  # noqa: E402
from optimization.optimizer import load_study  # noqa: E402
from optimization.search_spaces import SUGGEST_FUNCTIONS  # noqa: E402
from optimization.configs import FINAL_VALIDATION_CV  # noqa: E402
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
PHASE4_RAW = REPORTS / "phase4_raw"

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values
n = len(y)

MODELS = ["ExtraTrees", "CatBoost", "RandomForest", "GaussianProcess"]


def build_final_model(model_name: str, best_params: dict):
    fixed = optuna.trial.FixedTrial(best_params)
    return SUGGEST_FUNCTIONS[model_name](fixed)


# ===========================================================================
# Step 1: fresh per-repeat OOF matrices for the 4 SELECTED configurations
# ===========================================================================
oof = {}  # model -> (10, n) matrix
fold_rmse = {}  # model -> (50,) array in (repeat,fold) order

# ExtraTrees: reuse Phase 4 baseline checkpoint directly (identical config, identical protocol)
with open(PHASE4_RAW / "ExtraTrees.json") as f:
    et_baseline = json.load(f)
oof["ExtraTrees"] = np.array(et_baseline["oof_predictions"])
fold_rmse["ExtraTrees"] = np.array([r["rmse"] for r in sorted(et_baseline["fold_rows"], key=lambda r: (r["repeat"], r["fold"]))])
print(f"ExtraTrees: reused Phase 4 baseline OOF matrix, RMSE={fold_rmse['ExtraTrees'].mean():.3f}")

# CatBoost, RandomForest, GaussianProcess: fresh fit with Phase 5 tuned best params
for model_name in ["CatBoost", "RandomForest", "GaussianProcess"]:
    study = load_study(model_name)
    best_params = study.best_params
    reg_entry = MODEL_REGISTRY[model_name]
    oof_mat = np.full((10, n), np.nan)
    frmse = []
    t0 = time.perf_counter()
    for i, (train_idx, test_idx) in enumerate(FINAL_VALIDATION_CV.split(X)):
        repeat_idx, fold_idx = i // 5, i % 5
        model = build_final_model(model_name, best_params)
        pipe = build_pipeline(model, feature_set=reg_entry["feature_set"], scaler=reg_entry["scaler"])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds = pipe.predict(X.iloc[test_idx])
        oof_mat[repeat_idx, test_idx] = preds
        frmse.append(np.sqrt(np.mean((y[test_idx] - preds) ** 2)))
    oof[model_name] = oof_mat
    fold_rmse[model_name] = np.array(frmse)
    print(f"{model_name}: fresh OOF matrix built in {time.perf_counter()-t0:.1f}s, RMSE={np.mean(frmse):.3f}")

np.savez(REPORTS / "phase6_oof_matrices.npz", **{m: oof[m] for m in MODELS})
with open(REPORTS / "phase6_fold_rmse.json", "w") as f:
    json.dump({m: fold_rmse[m].tolist() for m in MODELS}, f, indent=2)

print("\nStep 1 complete: OOF matrices saved.")
