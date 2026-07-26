"""
Phase 5 analysis: for each of the 4 tuned models --
  1. Optuna study diagnostics (history, param importance, convergence, runtime distribution)
  2. Final, independent re-validation of the BEST hyperparameters under the full
     RepeatedKFold(5,10) protocol (identical to Phase 4's baseline benchmark) -- this is
     the number that actually gets compared to the Phase 4 baseline, never the Optuna
     objective's own (lighter-CV) best value.
  3. Paired statistical comparison: tuned vs. Phase 4 baseline (same fold sequence).
  4. Physical plausibility + residual diagnostics on the tuned models' OOF predictions.
  5. Diversity among the 4 tuned models (ensemble readiness for Phase 6).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models.registry import MODEL_REGISTRY  # noqa: E402
from models.metrics import compute_metrics  # noqa: E402
from models.evaluation import paired_comparison  # noqa: E402
from models.diagnostics import oof_point_estimates, residual_diagnostics, prediction_behavior  # noqa: E402
from preprocessing.pipelines import build_pipeline  # noqa: E402
from optimization.optimizer import load_study  # noqa: E402
from optimization.analysis import study_summary, convergence_check  # noqa: E402
from optimization.search_spaces import SUGGEST_FUNCTIONS  # noqa: E402
from optimization.configs import FINAL_VALIDATION_CV, N_TRIALS  # noqa: E402
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "tuned_pipelines"
FIG_DIR = REPORTS / "figures" / "phase5"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values
n = len(y)

MODELS = ["ExtraTrees", "CatBoost", "RandomForest", "GaussianProcess"]

# Phase 4 baseline results, for the paired comparison (re-derive fold_rows from checkpoints)
PHASE4_RAW = PROJECT_ROOT / "reports" / "phase4_raw"


def build_final_model(model_name: str, best_params: dict):
    """Reconstruct the tuned model from its best Optuna params using the same suggest_*
    logic, via a FixedTrial so every constructor argument (including derived ones like the
    GP kernel object) is built identically to how it was during search."""
    fixed = optuna.trial.FixedTrial(best_params)
    return SUGGEST_FUNCTIONS[model_name](fixed)


results = {}

for model_name in MODELS:
    print(f"\n=== {model_name} ===", flush=True)
    study = load_study(model_name)
    summary = study_summary(study)
    conv = convergence_check(summary["best_so_far_trajectory"])
    print(f"  best objective (light CV): {summary['best_value']:.3f}  n_complete={summary['n_complete']}  n_pruned={summary['n_pruned']}")
    print(f"  convergence: {conv['verdict']}")
    print(f"  top params by importance: {summary['param_importances']}")

    # -- Final re-validation: full RepeatedKFold(5,10), fresh model per fold --
    reg_entry = MODEL_REGISTRY[model_name]
    fold_rows = []
    oof = np.full((10, n), np.nan)
    fit_times = []
    t0 = time.perf_counter()
    for i, (train_idx, test_idx) in enumerate(FINAL_VALIDATION_CV.split(X)):
        repeat_idx, fold_idx = i // 5, i % 5
        model = build_final_model(model_name, study.best_params)
        pipe = build_pipeline(model, feature_set=reg_entry["feature_set"], scaler=reg_entry["scaler"])
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        tf0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - tf0)
        preds = pipe.predict(X_te)
        oof[repeat_idx, test_idx] = preds
        m = compute_metrics(y_te, preds)
        fold_rows.append({"repeat": repeat_idx, "fold": fold_idx, **m})
    total_time = time.perf_counter() - t0

    rmse_arr = np.array([r["rmse"] for r in fold_rows])
    tuned_summary = {
        "rmse_mean": float(rmse_arr.mean()), "rmse_std": float(rmse_arr.std()),
        "mae_mean": float(np.mean([r["mae"] for r in fold_rows])),
        "r2_mean": float(np.mean([r["r2"] for r in fold_rows])),
        "fit_time_mean": float(np.mean(fit_times)), "total_wall_time": total_time,
    }
    print(f"  FINAL re-validated RMSE (full RepeatedKFold(5,10)): {tuned_summary['rmse_mean']:.3f} +/- {tuned_summary['rmse_std']:.3f}")

    # -- Paired comparison vs. Phase 4 baseline (same fold sequence, so valid) --
    with open(PHASE4_RAW / f"{model_name}.json") as f:
        baseline = json.load(f)
    baseline_rmse = np.array([r["rmse"] for r in sorted(baseline["fold_rows"], key=lambda r: (r["repeat"], r["fold"]))])
    tuned_rmse_sorted = np.array([r["rmse"] for r in sorted(fold_rows, key=lambda r: (r["repeat"], r["fold"]))])
    cmp = paired_comparison(tuned_rmse_sorted, baseline_rmse)  # tuned - baseline
    print(f"  vs baseline ({baseline['summary']['rmse_mean']:.3f}): diff={cmp['mean_diff']:+.3f}  "
          f"paired_t_p={cmp['paired_t_p']:.4f}  wilcoxon_p={cmp['wilcoxon_p']:.4f}  "
          f"95% CI={cmp['bootstrap_ci95']}  significant={cmp['ci_excludes_zero']}")

    # -- Physical plausibility + residual diagnostics on tuned OOF predictions --
    pt = oof_point_estimates(oof)
    rdiag = residual_diagnostics(y, pt["mean"])
    pbeh = prediction_behavior(y, pt["mean"], pt["std"])

    # -- Serialize the tuned pipeline (fit on full data) --
    final_model = build_final_model(model_name, study.best_params)
    final_pipe = build_pipeline(final_model, feature_set=reg_entry["feature_set"], scaler=reg_entry["scaler"])
    final_pipe.fit(X, y)
    import joblib
    joblib.dump(final_pipe, ARTIFACTS / f"{model_name}_tuned_v1.joblib")

    results[model_name] = {
        "optuna_summary": summary, "convergence": conv,
        "tuned_final_validation": tuned_summary,
        "baseline_rmse_mean": baseline["summary"]["rmse_mean"],
        "vs_baseline": cmp,
        "residual_diagnostics": rdiag, "prediction_behavior": pbeh,
        "oof_mean_predictions": pt["mean"].tolist(),
    }

    # Convergence plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(summary["best_so_far_trajectory"])
    ax.set_title(f"{model_name}: best-objective-so-far (light-CV search)")
    ax.set_xlabel("completed trial index")
    ax.set_ylabel("RMSE (objective CV)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"convergence_{model_name}.png", dpi=130)
    plt.close(fig)

with open(REPORTS / "phase5_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Diversity among the 4 TUNED models
# ---------------------------------------------------------------------------
pred_df = pd.DataFrame({m: results[m]["oof_mean_predictions"] for m in MODELS})
diversity_corr = pred_df.corr()
resid_df = pd.DataFrame({m: y - np.array(results[m]["oof_mean_predictions"]) for m in MODELS})
error_corr = resid_df.corr()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(diversity_corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Tuned-model prediction correlation")
sns.heatmap(error_corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Tuned-model residual correlation")
plt.tight_layout()
plt.savefig(FIG_DIR / "tuned_model_diversity.png", dpi=130)
plt.close(fig)

with open(REPORTS / "phase5_diversity.json", "w") as f:
    json.dump({"prediction_corr": diversity_corr.round(4).to_dict(), "error_corr": error_corr.round(4).to_dict()}, f, indent=2)

print("\nPhase 5 analysis complete.")
