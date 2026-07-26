"""Phase 4: feature importance (impurity, permutation, SHAP) across model families,
to check whether different families agree on the physical drivers of yield."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, FEATURE_COLS, load_train  # noqa: E402
from models.pipelines import build_model_pipeline  # noqa: E402
from models.diagnostics import compute_importances  # noqa: E402
from preprocessing import config as cfg  # noqa: E402

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "phase4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split  # noqa: E402

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values
feature_names = cfg.FEATURE_SETS["core"]

# IMPORTANT: permutation importance must be evaluated on HELD-OUT data. ExtraTrees/CatBoost/
# RandomForest all reach near-zero in-sample RMSE (confirmed: ExtraTrees in-sample RMSE =
# 2.3e-13) -- computing permutation_importance on the same data the model was fit on gives
# a baseline score of ~0, so any permutation looks catastrophic and the resulting "importance"
# values are meaningless (tens of RMSE units, larger than the model's actual CV RMSE). Fit on
# an 80/20 split (matching the 5-fold CV's 20% holdout size) and evaluate importance on the
# held-out 20% instead.
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

MODELS_FOR_IMPORTANCE = ["ExtraTrees", "CatBoost", "RandomForest", "Ridge"]
results = {}

for name in MODELS_FOR_IMPORTANCE:
    print(f"[importance] {name} ...", flush=True)
    pipe = build_model_pipeline(name)
    pipe.fit(X_train, y_train)
    holdout_rmse = float(np.sqrt(np.mean((y_holdout - pipe.predict(X_holdout)) ** 2)))
    print(f"  holdout RMSE (80/20 split, single split -- not the CV estimate): {holdout_rmse:.2f}")
    imp = compute_importances(pipe, X_holdout, y_holdout, feature_names, raw_feature_names=list(FEATURE_COLS))
    imp["holdout_rmse_single_split"] = holdout_rmse
    results[name] = imp

    # SHAP: TreeExplainer for tree/boosting models, LinearExplainer for Ridge -- also
    # evaluated on the held-out split for the same reason.
    X_transformed = pipe[:-1].transform(X_holdout)  # everything except the final model step
    model = pipe.named_steps["model"]
    try:
        if name == "Ridge":
            explainer = shap.LinearExplainer(model, X_transformed)
        else:
            explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_transformed)
        mean_abs_shap = np.abs(sv).mean(axis=0)
        results[name]["shap_mean_abs"] = dict(zip(feature_names, mean_abs_shap.tolist()))
    except Exception as e:
        results[name]["shap_error"] = str(e)
        print(f"  SHAP failed for {name}: {e}")

with open(REPORTS / "phase4_importance.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Comparison figure: normalized permutation importance across models (raw features --
# what permutation_importance actually perturbs, since feature engineering is inside the pipeline)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = np.arange(len(FEATURE_COLS))
for i, name in enumerate(MODELS_FOR_IMPORTANCE):
    vals = np.array([results[name]["permutation_importance_mean"][f] for f in FEATURE_COLS])
    vals_norm = vals / (vals.max() if vals.max() > 0 else 1)
    ax.bar(x + i * width, vals_norm, width, label=name)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
ax.set_ylabel("Permutation importance (normalized to each model's max)")
ax.set_title("Feature importance agreement across model families")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance_comparison.png", dpi=130)
plt.close(fig)

print("Importance analysis complete.")
