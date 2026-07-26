"""
Phase 4 analysis: loads every completed models/checkpoint in reports/phase4_raw/*.json,
builds the leaderboard, runs statistical comparisons (paired t-test, Wilcoxon, Friedman,
Nemenyi, bootstrap CI), residual diagnostics, prediction-behavior/stability checks, and
model diversity (prediction correlation) analysis. Learning curves and feature importance
are handled in separate, smaller scripts (run_phase4_learning_curves.py,
run_phase4_importance.py) to keep this one fast and foreground-safe.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models.evaluation import fold_rmse_vector, paired_comparison, friedman_test, nemenyi_posthoc  # noqa: E402
from models.diagnostics import oof_point_estimates, residual_diagnostics, prediction_behavior  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "reports" / "phase4_raw"
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "phase4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")

train = load_train()
y = train[TARGET_COL].values
n = len(y)

# ---------------------------------------------------------------------------
# Load all successful checkpoints
# ---------------------------------------------------------------------------
loaded = {}
failed = {}
for f in sorted(RAW_DIR.glob("*.json")):
    with open(f) as fh:
        d = json.load(fh)
    if "error" in d:
        failed[d["model"]] = d["error"]
    else:
        loaded[d["model"]] = d

model_names = list(loaded.keys())
print(f"Loaded {len(model_names)} models: {model_names}")
print(f"Failed/skipped: {failed}")

results: dict = {"failed_models": failed}

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
leaderboard_rows = []
for name, d in loaded.items():
    s = d["summary"]
    leaderboard_rows.append({
        "model": name, "family": d["family"],
        "rmse_mean": s["rmse_mean"], "rmse_std": s["rmse_std"],
        "mae_mean": s["mae_mean"], "medae_mean": s["medae_mean"], "r2_mean": s["r2_mean"],
        "fit_time_mean": d["fit_time_mean"], "predict_time_mean": d["predict_time_mean"],
    })
leaderboard = pd.DataFrame(leaderboard_rows).sort_values("rmse_mean")
results["leaderboard"] = leaderboard.to_dict(orient="records")
print(leaderboard.to_string(index=False))

# ---------------------------------------------------------------------------
# Statistical comparison: Friedman across all models, then pairwise vs. the best model
# ---------------------------------------------------------------------------
rmse_matrix = np.column_stack([fold_rmse_vector(loaded[m]["fold_rows"]) for m in model_names])
friedman = friedman_test(rmse_matrix, model_names)
results["friedman"] = friedman
print(f"\nFriedman chi2={friedman['friedman_stat']:.3f}  p={friedman['friedman_p']:.2e}")

nemenyi = nemenyi_posthoc(rmse_matrix, model_names)
if nemenyi is not None:
    results["nemenyi_pvalues"] = nemenyi.to_dict()
    nemenyi.to_csv(REPORTS / "phase4_nemenyi_pvalues.csv")

best_model = leaderboard.iloc[0]["model"]
pairwise_rows = []
for name in model_names:
    if name == best_model:
        continue
    a = fold_rmse_vector(loaded[best_model]["fold_rows"])
    b = fold_rmse_vector(loaded[name]["fold_rows"])
    cmp = paired_comparison(b, a)  # b - a: positive means `name` worse than best
    pairwise_rows.append({"model": name, "vs_best": best_model, **cmp})
results["pairwise_vs_best"] = pairwise_rows

# ---------------------------------------------------------------------------
# Residual diagnostics + prediction behavior + stability (from OOF matrices)
# ---------------------------------------------------------------------------
diag_rows = []
oof_means = {}
for name, d in loaded.items():
    oof = np.array(d["oof_predictions"])
    pt = oof_point_estimates(oof)
    oof_means[name] = pt["mean"]
    rdiag = residual_diagnostics(y, pt["mean"])
    pbeh = prediction_behavior(y, pt["mean"], pt["std"])
    diag_rows.append({"model": name, **rdiag, **pbeh})
results["diagnostics"] = diag_rows

# ---------------------------------------------------------------------------
# Model diversity: correlation matrix of OOF predictions between models
# ---------------------------------------------------------------------------
pred_df = pd.DataFrame(oof_means)
diversity_corr = pred_df.corr(method="pearson")
results["diversity_correlation"] = diversity_corr.round(4).to_dict()

resid_df = pd.DataFrame({m: y - oof_means[m] for m in model_names})
error_diversity_corr = resid_df.corr(method="pearson")
results["error_diversity_correlation"] = error_diversity_corr.round(4).to_dict()

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.heatmap(diversity_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Prediction correlation across models")
sns.heatmap(error_diversity_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Error (residual) correlation across models")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_diversity.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------------------
# Residual diagnostic plots for top 6 models by RMSE
# ---------------------------------------------------------------------------
top6 = leaderboard.head(6)["model"].tolist()
fig, axes = plt.subplots(2, 6, figsize=(28, 9))
for i, name in enumerate(top6):
    pred = oof_means[name]
    resid = y - pred
    axes[0, i].scatter(pred, resid, alpha=0.6, s=18)
    axes[0, i].axhline(0, color="red", linewidth=1)
    axes[0, i].set_title(f"{name}\nresid vs pred")
    axes[0, i].set_xlabel("predicted")
    stats.probplot(resid, dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f"{name} QQ-plot")
plt.tight_layout()
plt.savefig(FIG_DIR / "residual_diagnostics_top6.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, 6, figsize=(28, 4.5))
for i, name in enumerate(top6):
    resid = y - oof_means[name]
    sns.histplot(resid, kde=True, ax=axes[i])
    axes[i].set_title(name)
    axes[i].axvline(0, color="red", linewidth=1)
plt.tight_layout()
plt.savefig(FIG_DIR / "residual_distributions_top6.png", dpi=130)
plt.close(fig)

with open(REPORTS / "phase4_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nPhase 4 analysis complete.")
print(f"Results: {REPORTS / 'phase4_analysis_results.json'}")
