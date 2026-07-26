"""Phase 6 analysis: diversity metrics, weighted averaging, blending, stacking,
region-specific performance, and the final statistical comparison vs. ExtraTrees alone."""
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
from features import build_feature_frame  # noqa: E402
from models.evaluation import paired_comparison  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
RNG_SEED = 42

train = load_train()
y = train[TARGET_COL].values
n = len(y)
avg_temp = build_feature_frame(train, ["avg_temp"])["avg_temp"].values

MODELS = ["ExtraTrees", "CatBoost", "RandomForest", "GaussianProcess"]

npz = np.load(REPORTS / "phase6_oof_matrices.npz")
oof = {m: npz[m] for m in MODELS}  # each (10, n)
with open(REPORTS / "phase6_fold_rmse.json") as f:
    fold_rmse = {m: np.array(v) for m, v in json.load(f).items()}

results = {}

# ===========================================================================
# Step 2: error diversity beyond prediction correlation
# ===========================================================================
oof_mean = {m: oof[m].mean(axis=0) for m in MODELS}
resid = {m: y - oof_mean[m] for m in MODELS}

zero_mask = y == 0.0
transition_mask = (avg_temp >= 410) & (avg_temp <= 480)
high_yield_mask = y > 50.0

diversity_rows = []
pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
for a, b in pairs:
    pred_corr = float(np.corrcoef(oof_mean[a], oof_mean[b])[0, 1])
    resid_corr = float(np.corrcoef(resid[a], resid[b])[0, 1])
    disagreement = np.abs(oof_mean[a] - oof_mean[b])
    # "Disagreement" = |pred_a - pred_b| > 10 percentage points (a physically meaningful
    # threshold given the target is a 0-100 percentage)
    disagree_rate = float((disagreement > 10).mean())
    disagree_zero = float((disagreement[zero_mask] > 10).mean()) if zero_mask.sum() else None
    disagree_transition = float((disagreement[transition_mask] > 10).mean()) if transition_mask.sum() else None
    disagree_high = float((disagreement[high_yield_mask] > 10).mean()) if high_yield_mask.sum() else None
    diversity_rows.append({
        "pair": f"{a}-{b}", "pred_corr": pred_corr, "resid_corr": resid_corr,
        "mean_abs_disagreement": float(disagreement.mean()),
        "disagreement_rate_gt10pts": disagree_rate,
        "disagreement_rate_zero_yield": disagree_zero,
        "disagreement_rate_transition_410_480K": disagree_transition,
        "disagreement_rate_high_yield": disagree_high,
    })
    print(f"{a}-{b}: pred_corr={pred_corr:.3f} resid_corr={resid_corr:.3f} "
          f"disagree>10pts: all={disagree_rate:.1%} zero={disagree_zero:.1%} "
          f"transition={disagree_transition:.1%} high={disagree_high:.1%}")
results["diversity"] = diversity_rows
results["region_counts"] = {
    "n_zero_yield": int(zero_mask.sum()), "n_transition_410_480K": int(transition_mask.sum()),
    "n_high_yield_gt50": int(high_yield_mask.sum()),
}

# ===========================================================================
# Step 3: weighted averaging (equal / RMSE-weighted / inverse-variance-weighted)
# fold-level RMSE computed directly on the OOF matrices -- weights are DETERMINISTIC
# functions of each model's already-known, already-validated mean RMSE/variance (from
# fold_rmse, itself out-of-fold), not fit on this data -- no leakage risk.
# ===========================================================================
mean_rmse = {m: fold_rmse[m].mean() for m in MODELS}
var_rmse = {m: fold_rmse[m].var() for m in MODELS}

def weighted_ensemble_fold_rmse(weights: dict) -> np.ndarray:
    w = np.array([weights[m] for m in MODELS])
    w = w / w.sum()
    combined_oof = sum(w[i] * oof[MODELS[i]] for i in range(len(MODELS)))  # (10, n)
    # Since the OOF matrix only preserves (repeat, row), not (repeat, fold) boundaries,
    # report per-repeat RMSE (10 values) instead of per-(repeat,fold) (50) for the combined
    # ensemble's own fold_rmse array -- still 10 independent, fully out-of-fold estimates.
    per_repeat_rmse = np.sqrt(np.mean((y[None, :] - combined_oof) ** 2, axis=1))
    return per_repeat_rmse

weight_schemes = {
    "equal": {m: 1.0 for m in MODELS},
    "rmse_weighted": {m: 1.0 / mean_rmse[m] for m in MODELS},
    "inverse_variance_weighted": {m: 1.0 / var_rmse[m] for m in MODELS},
}
weighting_rows = []
for name, weights in weight_schemes.items():
    w_norm = {m: weights[m] / sum(weights.values()) for m in MODELS}
    per_repeat_rmse = weighted_ensemble_fold_rmse(weights)
    weighting_rows.append({
        "scheme": name, "weights": w_norm,
        "rmse_mean": float(per_repeat_rmse.mean()), "rmse_std": float(per_repeat_rmse.std()),
        "per_repeat_rmse": per_repeat_rmse.tolist(),
    })
    print(f"[{name}] weights={ {k: round(v,3) for k,v in w_norm.items()} }  RMSE={per_repeat_rmse.mean():.3f}+/-{per_repeat_rmse.std():.3f}")
results["weighted_averaging"] = weighting_rows

# ===========================================================================
# Step 4/5: linear blender + stacking, trained on repeat-0 OOF, evaluated on repeats 1-9
# ===========================================================================
from sklearn.linear_model import Ridge, LinearRegression

def train_eval_meta(base_models: list[str], meta_cls, meta_kwargs: dict, name: str):
    X_train_meta = np.column_stack([oof[m][0] for m in base_models])  # repeat 0 -- (n, k)
    y_train_meta = y
    meta = meta_cls(**meta_kwargs).fit(X_train_meta, y_train_meta)

    eval_repeat_rmse = []
    for repeat in range(1, 10):
        X_eval = np.column_stack([oof[m][repeat] for m in base_models])
        pred = meta.predict(X_eval)
        eval_repeat_rmse.append(float(np.sqrt(np.mean((y - pred) ** 2))))
    eval_repeat_rmse = np.array(eval_repeat_rmse)
    print(f"[{name}] coef={dict(zip(base_models, meta.coef_.round(3))) if hasattr(meta,'coef_') else 'n/a'} "
          f"intercept={getattr(meta,'intercept_',None):.3f}  "
          f"eval(repeats 1-9) RMSE={eval_repeat_rmse.mean():.3f}+/-{eval_repeat_rmse.std():.3f}")
    return {
        "name": name, "base_models": base_models,
        "meta_coef": dict(zip(base_models, meta.coef_.tolist())) if hasattr(meta, "coef_") else None,
        "meta_intercept": float(meta.intercept_) if hasattr(meta, "intercept_") else None,
        "eval_rmse_mean": float(eval_repeat_rmse.mean()), "eval_rmse_std": float(eval_repeat_rmse.std()),
        "eval_per_repeat_rmse": eval_repeat_rmse.tolist(),
    }

blend_result = train_eval_meta(MODELS, LinearRegression, {}, "linear_blend_all4")
stack_result = train_eval_meta(["ExtraTrees", "CatBoost", "RandomForest"], Ridge, {"alpha": 1.0}, "stack_ET_CB_RF_ridge")
results["blending"] = blend_result
results["stacking"] = stack_result

# ===========================================================================
# Step 6: region-specific performance, single ExtraTrees vs. best ensemble
# ===========================================================================
best_ensemble_name = min(
    [(row["scheme"], row["rmse_mean"]) for row in weighting_rows] +
    [(blend_result["name"], blend_result["eval_rmse_mean"]), (stack_result["name"], stack_result["eval_rmse_mean"])],
    key=lambda t: t[1],
)[0]
print(f"\nBest ensemble by mean RMSE: {best_ensemble_name}")

if best_ensemble_name in weight_schemes:
    w = weight_schemes[best_ensemble_name]
    w_norm = {m: w[m] / sum(w.values()) for m in MODELS}
    best_ensemble_oof_mean = sum(w_norm[m] * oof_mean[m] for m in MODELS)
elif best_ensemble_name == blend_result["name"]:
    coefs = blend_result["meta_coef"]
    best_ensemble_oof_mean = sum(coefs[m] * oof_mean[m] for m in MODELS) + blend_result["meta_intercept"]
else:
    coefs = stack_result["meta_coef"]
    best_ensemble_oof_mean = sum(coefs[m] * oof_mean[m] for m in ["ExtraTrees", "CatBoost", "RandomForest"]) + stack_result["meta_intercept"]

region_rows = []
for region_name, mask in [("zero_yield", zero_mask), ("transition_410_480K", transition_mask), ("high_yield_gt50", high_yield_mask)]:
    if mask.sum() == 0:
        continue
    et_rmse = float(np.sqrt(np.mean((y[mask] - oof_mean["ExtraTrees"][mask]) ** 2)))
    ens_rmse = float(np.sqrt(np.mean((y[mask] - best_ensemble_oof_mean[mask]) ** 2)))
    region_rows.append({"region": region_name, "n": int(mask.sum()), "extratrees_rmse": et_rmse, "best_ensemble_rmse": ens_rmse, "diff": ens_rmse - et_rmse})
    print(f"[{region_name}] n={mask.sum()}  ExtraTrees={et_rmse:.3f}  {best_ensemble_name}={ens_rmse:.3f}  diff={ens_rmse-et_rmse:+.3f}")
results["region_specific"] = region_rows
results["best_ensemble_name"] = best_ensemble_name

# ===========================================================================
# Step 7: statistical comparison, ExtraTrees vs. best ensemble
# ===========================================================================
# Weighted-average ensembles: compare on the SAME 10 per-repeat values as ExtraTrees'
# own per-repeat RMSE (aggregate ExtraTrees' 50 fold values into 10 per-repeat means for
# a fair like-for-like paired comparison, since the weighted-ensemble RMSE above is
# per-repeat, not per-fold).
et_per_repeat = np.array([fold_rmse["ExtraTrees"][r * 5:(r + 1) * 5].mean() for r in range(10)])

if best_ensemble_name in weight_schemes:
    ens_per_repeat = np.array(next(row["per_repeat_rmse"] for row in weighting_rows if row["scheme"] == best_ensemble_name))
    cmp = paired_comparison(ens_per_repeat, et_per_repeat)
    comparison_note = "compared on 10 per-repeat RMSE values (weighted ensembles evaluated per-repeat, not per-fold)"
else:
    # blend/stack were evaluated on repeats 1-9 only (9 values)
    et_per_repeat_19 = et_per_repeat[1:]
    ens_key = "eval_per_repeat_rmse"
    ens_vals = np.array(blend_result[ens_key] if best_ensemble_name == blend_result["name"] else stack_result[ens_key])
    cmp = paired_comparison(ens_vals, et_per_repeat_19)
    comparison_note = "compared on 9 per-repeat RMSE values (repeats 1-9 only, since repeat 0 trained the meta-model)"

print(f"\n{best_ensemble_name} vs ExtraTrees: diff={cmp['mean_diff']:+.3f}  "
      f"t_p={cmp['paired_t_p']:.4f}  wilcoxon_p={cmp['wilcoxon_p']:.4f}  "
      f"CI={cmp['bootstrap_ci95']}  significant={cmp['ci_excludes_zero']}")
results["final_comparison"] = {**cmp, "note": comparison_note}

with open(REPORTS / "phase6_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nPhase 6 analysis complete.")
