"""Phase 4: learning curves for the top-tier models (bias/variance/data-sufficiency)."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models.pipelines import build_model_pipeline  # noqa: E402
from models.configs import RNG_SEED  # noqa: E402
from sklearn.model_selection import RepeatedKFold, learning_curve  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "phase4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values

TOP_MODELS = ["ExtraTrees", "CatBoost", "RandomForest", "GaussianProcess"]
train_sizes = np.linspace(0.2, 1.0, 8)

results = {}
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for i, name in enumerate(TOP_MODELS):
    pipe = build_model_pipeline(name)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RNG_SEED)  # lighter than main CV for speed
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, train_sizes=train_sizes, cv=cv,
        scoring="neg_root_mean_squared_error", random_state=RNG_SEED,
    )
    train_rmse = -train_scores
    val_rmse = -val_scores
    results[name] = {
        "train_sizes": sizes.tolist(),
        "train_rmse_mean": train_rmse.mean(axis=1).tolist(), "train_rmse_std": train_rmse.std(axis=1).tolist(),
        "val_rmse_mean": val_rmse.mean(axis=1).tolist(), "val_rmse_std": val_rmse.std(axis=1).tolist(),
    }
    ax = axes[i]
    ax.plot(sizes, train_rmse.mean(axis=1), "o-", label="train RMSE", color="#4C72B0")
    ax.fill_between(sizes, train_rmse.mean(axis=1) - train_rmse.std(axis=1), train_rmse.mean(axis=1) + train_rmse.std(axis=1), alpha=0.15, color="#4C72B0")
    ax.plot(sizes, val_rmse.mean(axis=1), "o-", label="validation RMSE", color="#C44E52")
    ax.fill_between(sizes, val_rmse.mean(axis=1) - val_rmse.std(axis=1), val_rmse.mean(axis=1) + val_rmse.std(axis=1), alpha=0.15, color="#C44E52")
    ax.set_title(name)
    ax.set_xlabel("training set size")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    print(f"{name}: train RMSE {train_rmse.mean(axis=1)[-1]:.2f} -> val RMSE {val_rmse.mean(axis=1)[-1]:.2f} at full size (gap={val_rmse.mean(axis=1)[-1]-train_rmse.mean(axis=1)[-1]:.2f})", flush=True)

plt.tight_layout()
plt.savefig(FIG_DIR / "learning_curves_top4.png", dpi=130)
plt.close(fig)

with open(REPORTS / "phase4_learning_curves.json", "w") as f:
    json.dump(results, f, indent=2)
print("Learning curves complete.")
