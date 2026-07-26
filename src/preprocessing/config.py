"""Centralized preprocessing configuration.

Phase 4 (and beyond) should change preprocessing behavior by editing/selecting entries
here, never by hard-coding feature lists or transformer choices in model code.
"""
from __future__ import annotations

from utils.io import FEATURE_COLS
from features import ALL_FEATURES, RECOMMENDED_FEATURES_CORE, RECOMMENDED_FEATURES_OPTIONAL

# ---------------------------------------------------------------------------
# Feature sets -- selectable by name, no code changes required to switch
# ---------------------------------------------------------------------------
FEATURE_SETS: dict[str, list[str]] = {
    "raw_only": list(FEATURE_COLS),
    "core": list(FEATURE_COLS) + list(RECOMMENDED_FEATURES_CORE),
    "core_plus_optional": list(FEATURE_COLS) + list(RECOMMENDED_FEATURES_CORE) + list(RECOMMENDED_FEATURES_OPTIONAL),
    "all_candidates": list(FEATURE_COLS) + list(ALL_FEATURES.keys()),
}
DEFAULT_FEATURE_SET = "core"

# ---------------------------------------------------------------------------
# Scaling options -- see scalers.py for the actual constructors
# ---------------------------------------------------------------------------
SCALER_NAMES = [
    "none", "standard", "robust", "minmax", "quantile_normal",
    "power_yeojohnson", "power_boxcox",
]

# ---------------------------------------------------------------------------
# Distribution-transform options -- see transformers.py; validity constraints
# (e.g. Box-Cox / log / sqrt / reciprocal require strictly positive input) are
# enforced there, not silently skipped here.
# ---------------------------------------------------------------------------
TRANSFORM_NAMES = ["none", "log1p", "sqrt", "reciprocal", "yeo_johnson", "box_cox"]

# Columns Phase 2/3 evidence flagged as candidates for a per-column distribution
# transform (see reports/preprocessing_report.md for the audit that justifies this list).
SKEW_CANDIDATE_COLUMNS = ["residence_proxy", "severity_index", "delta_T"]

# ---------------------------------------------------------------------------
# Outlier-handling options -- see transformers.py: Winsorizer / Clipper
# ---------------------------------------------------------------------------
OUTLIER_STRATEGIES = ["none", "winsorize_1_99", "clip_iqr"]

# ---------------------------------------------------------------------------
# Final Phase 3 recommendation, per model family -- backed by RepeatedKFold CV
# evidence in reports/preprocessing_report.md and reports/pipeline_benchmark_report.md.
# Phase 4 should read this dict rather than re-deriving preprocessing choices.
# ---------------------------------------------------------------------------
MODEL_FAMILY_PREPROCESSING: dict[str, dict] = {
    "Ridge": {"feature_set": "core", "scaler": "power_yeojohnson", "transform": "none", "outlier_strategy": "none"},
    "SVR": {"feature_set": "core", "scaler": "standard", "transform": "none", "outlier_strategy": "none"},
    "KNN": {"feature_set": "core", "scaler": "none", "transform": "none", "outlier_strategy": "none"},
    "GaussianProcess": {"feature_set": "core", "scaler": "power_yeojohnson", "transform": "none", "outlier_strategy": "none"},
    "RandomForest": {"feature_set": "core", "scaler": "none", "transform": "none", "outlier_strategy": "none"},
    "CatBoost": {"feature_set": "core", "scaler": "none", "transform": "none", "outlier_strategy": "none"},
}
