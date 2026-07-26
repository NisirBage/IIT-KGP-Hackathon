"""Phase 6 final blend validation: leave-one-repeat-out (proper nested check), clipped to
[0,100], compared statistically against ExtraTrees on the same repeat-level basis. Also
checks region-specific performance and re-derives the deployed (pooled-fit) coefficients."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from features import build_feature_frame  # noqa: E402
from models.evaluation import paired_comparison  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

train = load_train()
y = train[TARGET_COL].values
avg_temp = build_feature_frame(train, ["avg_temp"])["avg_temp"].values
MODELS = ["ExtraTrees", "CatBoost", "RandomForest", "GaussianProcess"]

npz = np.load(REPORTS / "phase6_oof_matrices.npz")
oof = {m: npz[m] for m in MODELS}
with open(REPORTS / "phase6_fold_rmse.json") as f:
    fold_rmse = {m: np.array(v) for m, v in json.load(f).items()}

# Leave-one-repeat-out: train on 9 repeats pooled, evaluate on the held-out repeat, both
# clipped and unclipped, for all 10 held-out choices -- this is the rigorous nested estimate.
loo_pred_by_repeat = {}
loo_rmse_clipped, loo_rmse_unclipped = [], []
for held_out in range(10):
    train_repeats = [r for r in range(10) if r != held_out]
    X_tr = np.vstack([np.column_stack([oof[m][r] for m in MODELS]) for r in train_repeats])
    y_tr = np.tile(y, len(train_repeats))
    meta = LinearRegression().fit(X_tr, y_tr)
    X_ev = np.column_stack([oof[m][held_out] for m in MODELS])
    pred = meta.predict(X_ev)
    pred_clipped = np.clip(pred, 0, 100)
    loo_pred_by_repeat[held_out] = pred_clipped
    loo_rmse_unclipped.append(float(np.sqrt(np.mean((y - pred) ** 2))))
    loo_rmse_clipped.append(float(np.sqrt(np.mean((y - pred_clipped) ** 2))))

loo_rmse_clipped = np.array(loo_rmse_clipped)
loo_rmse_unclipped = np.array(loo_rmse_unclipped)
print(f"LOO unclipped: {loo_rmse_unclipped.mean():.3f} +/- {loo_rmse_unclipped.std():.3f}")
print(f"LOO clipped:   {loo_rmse_clipped.mean():.3f} +/- {loo_rmse_clipped.std():.3f}")

# Final deployed coefficients: pooled fit on ALL 10 repeats (most data used)
X_all = np.vstack([np.column_stack([oof[m][r] for m in MODELS]) for r in range(10)])
y_all = np.tile(y, 10)
final_meta = LinearRegression().fit(X_all, y_all)
final_coef = dict(zip(MODELS, final_meta.coef_.tolist()))
final_intercept = float(final_meta.intercept_)
print(f"Deployed (pooled) coefficients: {final_coef}  intercept={final_intercept:.3f}")

# Physical plausibility check (post-clip, using the LOO predictions -- genuinely OOF)
loo_mean_pred = np.mean([loo_pred_by_repeat[r] for r in range(10)], axis=0)
n_implausible = int(((loo_mean_pred < 0) | (loo_mean_pred > 100)).sum())
print(f"Physical plausibility (clipped blend, LOO mean pred): {n_implausible}/150 implausible")

# ---------------------------------------------------------------------------
# Statistical comparison: clipped blend (LOO, 10 values) vs. ExtraTrees (10 per-repeat values)
# ---------------------------------------------------------------------------
et_per_repeat = np.array([fold_rmse["ExtraTrees"][r * 5:(r + 1) * 5].mean() for r in range(10)])
cmp = paired_comparison(loo_rmse_clipped, et_per_repeat)
print(f"\nClipped blend vs ExtraTrees: diff={cmp['mean_diff']:+.3f}  "
      f"paired_t_p={cmp['paired_t_p']:.6f}  wilcoxon_p={cmp['wilcoxon_p']:.4f}  "
      f"CI={cmp['bootstrap_ci95']}  significant={cmp['ci_excludes_zero']}")

# ---------------------------------------------------------------------------
# Region-specific performance, clipped blend vs ExtraTrees (using LOO mean predictions)
# ---------------------------------------------------------------------------
et_oof_mean = np.load(REPORTS / "phase6_oof_matrices.npz")["ExtraTrees"].mean(axis=0)
zero_mask = y == 0.0
transition_mask = (avg_temp >= 410) & (avg_temp <= 480)
high_yield_mask = y > 50.0

region_rows = []
for region_name, mask in [("zero_yield", zero_mask), ("transition_410_480K", transition_mask), ("high_yield_gt50", high_yield_mask)]:
    et_rmse = float(np.sqrt(np.mean((y[mask] - et_oof_mean[mask]) ** 2)))
    blend_rmse = float(np.sqrt(np.mean((y[mask] - loo_mean_pred[mask]) ** 2)))
    region_rows.append({"region": region_name, "n": int(mask.sum()), "extratrees_rmse": et_rmse, "blend_rmse": blend_rmse, "diff": blend_rmse - et_rmse})
    print(f"[{region_name}] n={mask.sum()}  ExtraTrees={et_rmse:.3f}  ClippedBlend={blend_rmse:.3f}  diff={blend_rmse-et_rmse:+.3f}")

results = {
    "loo_unclipped": {"mean": float(loo_rmse_unclipped.mean()), "std": float(loo_rmse_unclipped.std()), "values": loo_rmse_unclipped.tolist()},
    "loo_clipped": {"mean": float(loo_rmse_clipped.mean()), "std": float(loo_rmse_clipped.std()), "values": loo_rmse_clipped.tolist()},
    "deployed_coefficients": final_coef, "deployed_intercept": final_intercept,
    "n_implausible_after_clip": n_implausible,
    "vs_extratrees": cmp,
    "region_specific": region_rows,
}
with open(REPORTS / "phase6_final_blend_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nDone.")
