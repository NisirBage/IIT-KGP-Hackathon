"""
Phase 2: Physics-Informed Feature Engineering & Validation.

Computes, for every candidate engineered feature: correlation battery vs target,
incremental value vs its raw "parent" (nested F-test + nonlinear R^2 gain),
redundancy (correlation matrix, VIF, hierarchical clustering), bootstrap stability
of correlation/MI rankings, nonlinearity shape diagnostics (linear vs quadratic R^2,
LOWESS), a zero-yield separability study using only engineered features, and a
lightweight (no tuning) benchmark of Ridge / RandomForest / CatBoost across three
feature sets. Everything is dumped to JSON so the written reports quote real numbers.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import FEATURE_COLS, TARGET_COL, load_train  # noqa: E402
from features import ALL_FEATURES, build_feature_frame  # noqa: E402
from features.build import PARENT_OF  # noqa: E402

import dcor
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

sns.set_theme(style="whitegrid", context="notebook")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "phase2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

train = load_train()
y = train[TARGET_COL].values
n = len(y)

engineered = build_feature_frame(train)
full = pd.concat([train, engineered], axis=1)

results: dict = {}

# ===========================================================================
# 4. FEATURE VALIDATION: correlation battery for every engineered feature
# ===========================================================================
corr_rows = []
for name in ALL_FEATURES:
    x = engineered[name].values
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kend = stats.kendalltau(x, y)
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=RNG_SEED)[0]
    dc = float(dcor.distance_correlation(x, y))
    corr_rows.append({
        "feature": name,
        "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue),
        "kendall_tau": float(kend.statistic), "kendall_p": float(kend.pvalue),
        "mutual_info": float(mi), "distance_corr": dc,
    })
results["feature_validation"] = corr_rows

# ===========================================================================
# 5. INCREMENTAL VALUE: nested linear F-test + nonlinear (kNN) R2 gain +
#    sequential feature selection over the full candidate set
# ===========================================================================

def r2_of(X: np.ndarray, yy: np.ndarray) -> float:
    pred = LinearRegression().fit(X, yy).predict(X)
    ss_res = ((yy - pred) ** 2).sum()
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def partial_f_test(parent_cols: list[str], feature_col: str) -> dict:
    Xp = full[parent_cols].values
    Xpf = full[parent_cols + [feature_col]].values
    r2p = r2_of(Xp, y)
    r2pf = r2_of(Xpf, y)
    p1, p2 = Xp.shape[1], Xpf.shape[1]
    f_stat = ((r2pf - r2p) / (p2 - p1)) / ((1 - r2pf) / (n - p2 - 1))
    f_p = float(1 - stats.f.cdf(f_stat, p2 - p1, n - p2 - 1))
    return {"r2_parent": r2p, "r2_parent_plus_feature": r2pf, "f_stat": float(f_stat), "f_p": f_p}


def knn_cv_r2(X: np.ndarray, yy: np.ndarray, k: int = 10) -> float:
    """5-fold CV R^2 of a simple kNN regressor -- nonlinear/nonparametric proxy for
    'does this feature set carry more information', complementing the linear F-test."""
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    scores = cross_val_score(
        KNeighborsRegressor(n_neighbors=k), Xs, yy,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=RNG_SEED), scoring="r2",
    )
    return float(np.mean(scores))


incremental_rows = []
for name, parents in PARENT_OF.items():
    valid_parents = [p for p in parents if p in full.columns]
    if not valid_parents:
        continue
    ftest = partial_f_test(valid_parents, name)
    knn_parent = knn_cv_r2(full[valid_parents].values, y)
    knn_parent_plus = knn_cv_r2(full[valid_parents + [name]].values, y)
    incremental_rows.append({
        "feature": name, "parents": valid_parents,
        **ftest,
        "knn_cv_r2_parent_only": knn_parent,
        "knn_cv_r2_parent_plus_feature": knn_parent_plus,
        "knn_gain": knn_parent_plus - knn_parent,
    })
results["incremental_value"] = incremental_rows

# Sequential forward feature selection over the FULL candidate set (raw + engineered),
# scored by 5x3 repeated CV R^2 with Ridge (fast, well-behaved with correlated features)
all_candidate_cols = FEATURE_COLS + list(ALL_FEATURES.keys())
X_all = full[all_candidate_cols].values
X_all_std = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)

sfs = SequentialFeatureSelector(
    Ridge(alpha=1.0), n_features_to_select=8, direction="forward",
    scoring="neg_root_mean_squared_error",
    cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=RNG_SEED),
)
sfs.fit(X_all_std, y)
selected_mask = sfs.get_support()
sfs_selected = [c for c, m in zip(all_candidate_cols, selected_mask) if m]
results["sequential_feature_selection"] = {"selected_features_in_order_found": sfs_selected}

# ===========================================================================
# 6. REDUNDANCY ANALYSIS: correlation matrix, VIF, hierarchical clustering
# ===========================================================================
engineered_corr = engineered.corr(method="spearman")
results["redundancy_correlation_matrix"] = engineered_corr.round(4).to_dict()

# VIF on the full engineered set (standardized to avoid scale issues)
X_eng = engineered.values
X_eng_std = (X_eng - X_eng.mean(axis=0)) / X_eng.std(axis=0)
vif_rows = []
for i, name in enumerate(engineered.columns):
    try:
        vif = variance_inflation_factor(X_eng_std, i)
    except Exception:
        vif = float("inf")
    vif_rows.append({"feature": name, "vif": float(vif)})
results["vif"] = vif_rows

# Hierarchical clustering on (1 - |spearman corr|) distance
dist = 1 - engineered_corr.abs().values
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2  # enforce exact symmetry against float round-off
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method="average")
cluster_labels = fcluster(Z, t=0.3, criterion="distance")  # merge features >=0.7 |corr|
clusters: dict[int, list[str]] = {}
for name, lbl in zip(engineered.columns, cluster_labels):
    clusters.setdefault(int(lbl), []).append(name)
results["hierarchical_clusters_at_0.7_abs_corr"] = clusters

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, labels=engineered.columns.tolist(), ax=ax, leaf_rotation=90)
ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="cluster cut (|corr|=0.7)")
ax.set_title("Feature dendrogram (1 - |Spearman corr| distance)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_dendrogram.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(engineered_corr, cmap="vlag", center=0, annot=False, ax=ax)
ax.set_title("Engineered feature correlation matrix (Spearman)")
plt.tight_layout()
plt.savefig(FIG_DIR / "engineered_correlation_heatmap.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 7. STABILITY ANALYSIS: bootstrap correlation / MI ranking stability
# ===========================================================================
N_BOOT = 500
rng = np.random.default_rng(RNG_SEED)
boot_spearman = {name: [] for name in ALL_FEATURES}
boot_mi = {name: [] for name in ALL_FEATURES}
boot_ranks = []

for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    yb = y[idx]
    row_spear = {}
    row_mi = {}
    for name in ALL_FEATURES:
        xb = engineered[name].values[idx]
        s = stats.spearmanr(xb, yb).statistic
        boot_spearman[name].append(s)
        row_spear[name] = abs(s)
    # cheap MI proxy every 10th bootstrap only (MI is the slow step)
    if b % 10 == 0:
        for name in ALL_FEATURES:
            xb = engineered[name].values[idx]
            mi = mutual_info_regression(xb.reshape(-1, 1), yb, random_state=RNG_SEED)[0]
            boot_mi[name].append(mi)
            row_mi[name] = mi
    ranked = sorted(row_spear.items(), key=lambda kv: -kv[1])
    boot_ranks.append([name for name, _ in ranked])

stability_rows = []
for name in ALL_FEATURES:
    sp = np.array(boot_spearman[name])
    mi_vals = np.array(boot_mi[name]) if boot_mi[name] else np.array([np.nan])
    # rank stability: how often this feature lands in the top-8 across bootstraps
    top8_freq = np.mean([name in ranks[:8] for ranks in boot_ranks])
    stability_rows.append({
        "feature": name,
        "spearman_mean": float(np.mean(sp)), "spearman_std": float(np.std(sp)),
        "mi_mean": float(np.mean(mi_vals)), "mi_std": float(np.std(mi_vals)),
        "pct_bootstraps_in_top8_by_|spearman|": float(top8_freq),
    })
results["stability"] = stability_rows

# ===========================================================================
# 8. NONLINEARITY INVESTIGATION: linear vs quadratic R^2 + LOWESS for top features
# ===========================================================================
shape_rows = []
for name in ALL_FEATURES:
    x = engineered[name].values
    r2_lin = r2_of(x.reshape(-1, 1), y)
    r2_quad = r2_of(np.column_stack([x, x ** 2]), y)
    shape_rows.append({
        "feature": name, "r2_linear": r2_lin, "r2_quadratic": r2_quad,
        "quadratic_gain": r2_quad - r2_lin,
    })
results["shape_diagnostics"] = shape_rows

top_by_dcor = sorted(corr_rows, key=lambda r: -r["distance_corr"])[:8]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
for i, row in enumerate(top_by_dcor):
    name = row["feature"]
    x = engineered[name].values
    order = np.argsort(x)
    sm = lowess(y, x, frac=0.6, return_sorted=True)
    axes[i].scatter(x, y, alpha=0.5, s=20)
    axes[i].plot(sm[:, 0], sm[:, 1], color="red", linewidth=2, label="LOWESS")
    axes[i].set_title(f"{name}\n(dcor={row['distance_corr']:.3f})")
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "nonlinearity_lowess_top8.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 9. ZERO-YIELD MECHANISM: separability using ONLY engineered physics features
# ===========================================================================
zero_target = (y == 0.0).astype(int)
engineered_physics_cols = [
    "avg_temp", "residence_proxy", "delta_T", "severity_index",
    "max_temp_approx", "abs_delta_T",
]
Xz = engineered[engineered_physics_cols].values
Xz_std = (Xz - Xz.mean(axis=0)) / Xz.std(axis=0)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)  # regression-style splitter reused for stratified-like folds via label shuffling is unnecessary here; use classifier scoring directly
from sklearn.model_selection import RepeatedStratifiedKFold
cv_strat = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)

zero_yield_rows = []
for clf_name, clf in [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("DecisionTree(depth<=3)", DecisionTreeClassifier(max_depth=3, random_state=RNG_SEED)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.2)),
]:
    acc = cross_val_score(clf, Xz_std, zero_target, cv=cv_strat, scoring="accuracy")
    auc = cross_val_score(clf, Xz_std, zero_target, cv=cv_strat, scoring="roc_auc")
    zero_yield_rows.append({
        "model": clf_name,
        "cv_accuracy_mean": float(np.mean(acc)), "cv_accuracy_std": float(np.std(acc)),
        "cv_auc_mean": float(np.mean(auc)), "cv_auc_std": float(np.std(auc)),
    })
results["zero_yield_separability"] = {
    "features_used": engineered_physics_cols,
    "baseline_majority_class_accuracy": float(1 - zero_target.mean()),
    "models": zero_yield_rows,
}

# fit one decision tree on full data (not for deployment -- for the printed rule only)
tree = DecisionTreeClassifier(max_depth=3, random_state=RNG_SEED).fit(Xz, zero_target)
from sklearn.tree import export_text
tree_rules = export_text(tree, feature_names=engineered_physics_cols)
results["zero_yield_tree_rules"] = tree_rules

# ===========================================================================
# 10. LIGHTWEIGHT BENCHMARKING (no hyperparameter tuning): Ridge / RF / CatBoost
#     across Raw / Raw+Validated / Raw+All feature sets, identical repeated CV
# ===========================================================================
from catboost import CatBoostRegressor

VALIDATED_FEATURES = [
    "avg_temp", "residence_proxy", "severity_index", "delta_T", "residence_sq",
]

feature_sets = {
    "raw_only": FEATURE_COLS,
    "raw_plus_validated": FEATURE_COLS + VALIDATED_FEATURES,
    "raw_plus_all_candidates": FEATURE_COLS + list(ALL_FEATURES.keys()),
}

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RNG_SEED),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=RNG_SEED),
}

bench_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)
benchmark_rows = []
for set_name, cols in feature_sets.items():
    X = full[cols].values
    for model_name, model in models.items():
        if model_name == "Ridge":
            Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        else:
            Xs = X
        scores = cross_val_score(model, Xs, y, cv=bench_cv, scoring="neg_root_mean_squared_error")
        rmse = -scores
        benchmark_rows.append({
            "feature_set": set_name, "n_features": len(cols), "model": model_name,
            "rmse_mean": float(np.mean(rmse)), "rmse_std": float(np.std(rmse)),
        })
        print(f"{set_name:28s} {model_name:14s} RMSE={np.mean(rmse):7.3f} +/- {np.std(rmse):.3f}")
results["benchmark"] = benchmark_rows

# ===========================================================================
# Dump
# ===========================================================================
with open(REPORTS / "phase2_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nPhase 2 analysis complete.")
print(f"Figures: {FIG_DIR}")
print(f"Results JSON: {REPORTS / 'phase2_analysis_results.json'}")
