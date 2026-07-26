"""
Phase 1A: Dataset Integrity & Evidence-Driven EDA.

Runs the full audit described in reports/ (dataset passport, zero-yield
verification, train/test distribution shift, statistical group tests,
correlation analysis, and diagnostic figures) and writes every number it
reports directly into the generated markdown files, so nothing here is
hand-typed / hand-fabricated after the fact.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import FEATURE_COLS, TARGET_COL, load_test, load_train  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

try:
    import dcor
    HAS_DCOR = True
except ImportError:
    HAS_DCOR = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

# ---------------------------------------------------------------------------
# Load data + engineered candidate features (for correlation / registry only
# -- NOT fed into any model in this phase)
# ---------------------------------------------------------------------------
train = load_train()
test = load_test()

def add_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["residence_proxy"] = df["length_m"] / df["flow_rate_L_min"]
    df["delta_T_jacket_inlet"] = df["jacket_temperature_K"] - df["inlet_temperature_K"]
    df["avg_temp"] = (df["jacket_temperature_K"] + df["inlet_temperature_K"]) / 2.0
    df["severity_index"] = df["residence_proxy"] * df["delta_T_jacket_inlet"]
    df["inv_inlet_temp"] = 1.0 / df["inlet_temperature_K"]
    return df

train_ext = add_candidates(train)
test_ext = add_candidates(test)
CANDIDATE_COLS = ["residence_proxy", "delta_T_jacket_inlet", "avg_temp", "severity_index", "inv_inlet_temp"]

results: dict = {}

# ===========================================================================
# 1. DATASET PASSPORT
# ===========================================================================

def decimal_places(series: pd.Series, max_check: int = 6) -> int:
    """Infer the max number of meaningful decimal places in a float column."""
    def dp(x: float) -> int:
        s = f"{x:.{max_check}f}".rstrip("0")
        if "." not in s:
            return 0
        return len(s.split(".")[1])
    return int(series.dropna().map(dp).max())


def passport_for(df: pd.DataFrame, name: str, has_target: bool) -> dict:
    cols = FEATURE_COLS + ([TARGET_COL] if has_target else [])
    d = {
        "name": name,
        "n_samples": len(df),
        "n_features": len(FEATURE_COLS),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in cols},
        "n_unique": {c: int(df[c].nunique()) for c in cols},
        "constant_columns": [c for c in cols if df[c].nunique() <= 1],
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_feature_vectors": int(df[FEATURE_COLS].duplicated().sum()),
        "decimal_places": {c: decimal_places(df[c]) for c in cols},
        "pct_zero": {c: float((df[c] == 0).mean() * 100) for c in cols},
    }
    if has_target:
        d["duplicated_targets"] = int(df[TARGET_COL].duplicated().sum())
        d["target_out_of_range_0_100"] = int(((df[TARGET_COL] < 0) | (df[TARGET_COL] > 100)).sum())
    # suspicious / impossible physical values
    suspicious = []
    if (df["flow_rate_L_min"] <= 0).any():
        suspicious.append(f"flow_rate_L_min <= 0: {int((df['flow_rate_L_min'] <= 0).sum())} rows")
    if (df["concentration_mol_L"] <= 0).any():
        suspicious.append(f"concentration_mol_L <= 0: {int((df['concentration_mol_L'] <= 0).sum())} rows")
    if (df["length_m"] <= 0).any():
        suspicious.append(f"length_m <= 0: {int((df['length_m'] <= 0).sum())} rows")
    if (df["inlet_temperature_K"] <= 0).any():
        suspicious.append(f"inlet_temperature_K <= 0 (below absolute zero context): {int((df['inlet_temperature_K'] <= 0).sum())} rows")
    if (df["jacket_temperature_K"] <= 0).any():
        suspicious.append(f"jacket_temperature_K <= 0: {int((df['jacket_temperature_K'] <= 0).sum())} rows")
    d["suspicious_values"] = suspicious
    return d


train_passport = passport_for(train, "train_dataset.csv", has_target=True)
test_passport = passport_for(test, "test_dataset.csv", has_target=False)
results["passport"] = {"train": train_passport, "test": test_passport}

# ===========================================================================
# 2. ZERO-YIELD VERIFICATION
# ===========================================================================
y = train[TARGET_COL].values
n = len(y)
exact_zero = int((y == 0.0).sum())
lt_0001 = int(((y > 0) & (y < 0.001)).sum())
lt_001 = int(((y > 0) & (y < 0.01)).sum())
lt_01 = int(((y > 0) & (y < 0.1)).sum())
lt_1 = int(((y > 0) & (y < 1.0)).sum())
sorted_nonzero = np.sort(y[y > 0])
smallest_nonzero = float(sorted_nonzero[0]) if len(sorted_nonzero) else None

# GMM-based modality check: fit 1 vs 2 vs 3 component GMMs to the FULL target
# and to the NONZERO-only subset, compare BIC.
def gmm_bic_scan(data: np.ndarray, max_k: int = 3) -> dict:
    data = data.reshape(-1, 1)
    out = {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=RNG_SEED, n_init=5)
        gmm.fit(data)
        out[k] = float(gmm.bic(data))
    return out

bic_full = gmm_bic_scan(y)
bic_nonzero = gmm_bic_scan(y[y > 0]) if (y > 0).sum() >= 6 else {}

results["zero_yield"] = {
    "n_total": n,
    "exact_zero": exact_zero,
    "pct_exact_zero": round(exact_zero / n * 100, 2),
    "nonzero_lt_0.001": lt_0001,
    "nonzero_lt_0.01": lt_001,
    "nonzero_lt_0.1": lt_01,
    "nonzero_lt_1.0": lt_1,
    "smallest_nonzero_value": smallest_nonzero,
    "gmm_bic_full_target": bic_full,
    "gmm_bic_nonzero_only": bic_nonzero,
}

# Figure: full distribution + zoomed near-zero + log-scale
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.histplot(y, bins=30, ax=axes[0], color="#4C72B0")
axes[0].set_title("overall_yield — full distribution")
axes[0].set_xlabel("overall_yield")

sns.histplot(y[y < 5], bins=30, ax=axes[1], color="#C44E52")
axes[1].set_title("overall_yield — zoomed to [0, 5)")
axes[1].set_xlabel("overall_yield")

axes[2].hist(y, bins=30, color="#55A868")
axes[2].set_yscale("log")
axes[2].set_title("overall_yield — log-scale y axis")
axes[2].set_xlabel("overall_yield")
plt.tight_layout()
plt.savefig(FIG_DIR / "target_distribution.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 3. TRAIN vs TEST DISTRIBUTION COMPARISON
# ===========================================================================

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using train-defined quantile bins."""
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))
    if len(cut_points) < 3:
        return float("nan")
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf
    e_counts, _ = np.histogram(expected, bins=cut_points)
    a_counts, _ = np.histogram(actual, bins=cut_points)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


train_test_rows = []
for col in FEATURE_COLS:
    tr = train[col].values
    te = test[col].values
    ks_stat, ks_p = stats.ks_2samp(tr, te)
    wass = stats.wasserstein_distance(tr, te)
    psi_val = psi(tr, te)
    train_test_rows.append({
        "feature": col,
        "train_mean": float(np.mean(tr)), "test_mean": float(np.mean(te)),
        "train_std": float(np.std(tr, ddof=1)), "test_std": float(np.std(te, ddof=1)),
        "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        "wasserstein": float(wass), "psi": psi_val,
    })
results["train_test_comparison"] = train_test_rows

# Figure grid: hist+kde, boxplot, ecdf for each feature (train vs test)
fig, axes = plt.subplots(len(FEATURE_COLS), 3, figsize=(15, 4 * len(FEATURE_COLS)))
for i, col in enumerate(FEATURE_COLS):
    sns.histplot(train[col], color="#4C72B0", label="train", kde=True, stat="density", alpha=0.5, ax=axes[i, 0])
    sns.histplot(test[col], color="#DD8452", label="test", kde=True, stat="density", alpha=0.5, ax=axes[i, 0])
    axes[i, 0].set_title(f"{col}: hist+KDE")
    axes[i, 0].legend()

    combined = pd.concat([
        pd.DataFrame({col: train[col], "set": "train"}),
        pd.DataFrame({col: test[col], "set": "test"}),
    ])
    sns.boxplot(data=combined, x="set", y=col, ax=axes[i, 1])
    axes[i, 1].set_title(f"{col}: boxplot")

    for data, label, color in [(train[col], "train", "#4C72B0"), (test[col], "test", "#DD8452")]:
        xs = np.sort(data.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[i, 2].step(xs, ys, label=label, color=color)
    axes[i, 2].set_title(f"{col}: ECDF")
    axes[i, 2].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "train_test_comparison.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 4. STATISTICAL VALIDATION: zero-yield vs non-zero-yield groups
# ===========================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = RNG_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


zero_mask = train_ext[TARGET_COL] == 0.0
group_zero = train_ext[zero_mask]
group_nonzero = train_ext[~zero_mask]

group_test_cols = FEATURE_COLS + CANDIDATE_COLS
group_stat_rows = []
for col in group_test_cols:
    a = group_zero[col].values
    b = group_nonzero[col].values
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    delta = cliffs_delta(a, b)
    ci_lo, ci_hi = bootstrap_ci_mean_diff(a, b)
    group_stat_rows.append({
        "feature": col,
        "mean_zero_group": float(np.mean(a)), "mean_nonzero_group": float(np.mean(b)),
        "welch_t": float(t_stat), "welch_p": float(t_p),
        "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
        "cohens_d": d, "cliffs_delta": delta,
        "mean_diff_ci95": [ci_lo, ci_hi],
    })
results["zero_vs_nonzero_group_stats"] = group_stat_rows

# Violin plots for group comparison
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()
for i, col in enumerate(group_test_cols):
    plot_df = train_ext[[col]].copy()
    plot_df["group"] = np.where(zero_mask, "zero-yield", "non-zero")
    sns.violinplot(data=plot_df, x="group", y=col, ax=axes[i], inner="quartile")
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig(FIG_DIR / "zero_vs_nonzero_violins.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 5. CORRELATION ANALYSIS (raw features + candidates vs target)
# ===========================================================================
corr_rows = []
for col in FEATURE_COLS + CANDIDATE_COLS:
    x = train_ext[col].values
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=RNG_SEED)[0]
    dcorr = float(dcor.distance_correlation(x, y)) if HAS_DCOR else None
    corr_rows.append({
        "feature": col,
        "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue),
        "kendall_tau": float(kendall.statistic), "kendall_p": float(kendall.pvalue),
        "mutual_info": float(mi),
        "distance_corr": dcorr,
    })
results["correlations"] = corr_rows

# Feature-feature collinearity (raw features only)
feat_corr_matrix = train[FEATURE_COLS].corr(method="spearman")
results["feature_collinearity_spearman"] = feat_corr_matrix.round(3).to_dict()

fig, ax = plt.subplots(figsize=(7, 6))
full_corr = train_ext[FEATURE_COLS + [TARGET_COL]].corr(method="spearman")
sns.heatmap(full_corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Spearman correlation heatmap (raw features + target)")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=130)
plt.close(fig)

# ===========================================================================
# 6. VISUAL EXPLORATION: pairplot, target-vs-feature, interaction plot, outliers
# ===========================================================================
pp = sns.pairplot(
    train_ext[FEATURE_COLS + [TARGET_COL]],
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "s": 25},
)
pp.figure.suptitle("Pairplot: raw features + target", y=1.02)
pp.savefig(FIG_DIR / "pairplot.png", dpi=130)
plt.close("all")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    sc = axes[i].scatter(train[col], y, c=zero_mask.map({True: "#C44E52", False: "#4C72B0"}), alpha=0.7, s=20)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("overall_yield")
    axes[i].set_title(f"yield vs {col}")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "target_vs_features.png", dpi=130)
plt.close(fig)

# Key interaction plot: residence_proxy vs delta_T_jacket_inlet, colored by yield
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(
    train_ext["residence_proxy"], train_ext["delta_T_jacket_inlet"],
    c=train_ext[TARGET_COL], cmap="viridis", s=45, edgecolor="k", linewidth=0.3,
)
plt.colorbar(sc, label="overall_yield")
ax.set_xlabel("residence_proxy = length_m / flow_rate_L_min")
ax.set_ylabel("delta_T_jacket_inlet = jacket_T - inlet_T")
ax.set_title("Interaction: residence time proxy x net thermal driving force")
plt.tight_layout()
plt.savefig(FIG_DIR / "interaction_residence_deltaT.png", dpi=130)
plt.close(fig)

# Outlier diagnostics (IQR method) for raw features, train+test
outlier_rows = []
for name, df in [("train", train), ("test", test)]:
    for col in FEATURE_COLS:
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        outlier_rows.append({"dataset": name, "feature": col, "n_outliers_iqr": n_out, "lower": float(lo), "upper": float(hi)})
results["outliers_iqr"] = outlier_rows

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, col in enumerate(FEATURE_COLS):
    combined = pd.concat([
        pd.DataFrame({col: train[col], "set": "train"}),
        pd.DataFrame({col: test[col], "set": "test"}),
    ])
    sns.boxplot(data=combined, x="set", y=col, ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig(FIG_DIR / "outlier_boxplots.png", dpi=130)
plt.close(fig)

# ===========================================================================
# Dump raw results as JSON for downstream markdown generation / audit trail
# ===========================================================================
with open(REPORTS / "phase1_eda_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Phase 1A EDA computation complete.")
print(f"HAS_DCOR = {HAS_DCOR}")
print(f"Figures written to: {FIG_DIR}")
print(f"Results JSON: {REPORTS / 'phase1_eda_results.json'}")
