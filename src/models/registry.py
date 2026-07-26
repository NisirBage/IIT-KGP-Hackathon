"""Model registry: every candidate model is instantiated from here, paired with the
preprocessing settings it should run under (extending preprocessing.config for the models
Phase 3 didn't cover). No model should be constructed ad hoc anywhere else in Phase 4.

Preprocessing assignment for models Phase 3 did not directly benchmark is a documented
extrapolation from Phase 3 evidence, not independently re-tested:
- Linear-family models (LinearRegression, Lasso, ElasticNet) get the same scaler as Ridge
  (power_yeojohnson) -- all are linear models with the same scale sensitivity profile.
- Additional tree ensembles (ExtraTrees, HistGradientBoosting, XGBoost, LightGBM) and
  boosting methods with tree base learners (NGBoost, EBM) get "none", extrapolating the
  scale-invariance Phase 3 empirically confirmed (to 3 decimal places) for RandomForest and
  CatBoost specifically.
"""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

RNG_SEED = 42

# ---------------------------------------------------------------------------
# (model_name -> (factory(), preprocessing_settings, family))
# preprocessing_settings: {"scaler": ..., "feature_set": "core"}
# ---------------------------------------------------------------------------

def _catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(verbose=False, random_state=RNG_SEED)


def _xgboost():
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=RNG_SEED, verbosity=0)


def _lightgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=RNG_SEED, verbosity=-1)


def _ngboost():
    from ngboost import NGBRegressor
    return NGBRegressor(random_state=RNG_SEED, verbose=False)


def _ebm():
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(random_state=RNG_SEED)


MODEL_REGISTRY: dict[str, dict] = {
    "LinearRegression": {
        "factory": lambda: LinearRegression(),
        "family": "Linear", "scaler": "power_yeojohnson", "feature_set": "core",
    },
    "Ridge": {
        "factory": lambda: Ridge(alpha=1.0, random_state=RNG_SEED),
        "family": "Linear", "scaler": "power_yeojohnson", "feature_set": "core",
    },
    "Lasso": {
        "factory": lambda: Lasso(alpha=1.0, random_state=RNG_SEED),
        "family": "Linear", "scaler": "power_yeojohnson", "feature_set": "core",
    },
    "ElasticNet": {
        "factory": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RNG_SEED),
        "family": "Linear", "scaler": "power_yeojohnson", "feature_set": "core",
    },
    "SVR_RBF": {
        "factory": lambda: SVR(kernel="rbf"),
        "family": "Kernel", "scaler": "standard", "feature_set": "core",
    },
    "GaussianProcess": {
        "factory": lambda: GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=RNG_SEED, normalize_y=True),
        "family": "Kernel", "scaler": "power_yeojohnson", "feature_set": "core",
    },
    "KNN": {
        "factory": lambda: KNeighborsRegressor(n_neighbors=10),
        "family": "Instance-based", "scaler": "none", "feature_set": "core",
    },
    "RandomForest": {
        "factory": lambda: RandomForestRegressor(n_estimators=300, random_state=RNG_SEED),
        "family": "Tree Ensemble", "scaler": "none", "feature_set": "core",
    },
    "ExtraTrees": {
        "factory": lambda: ExtraTreesRegressor(n_estimators=300, random_state=RNG_SEED),
        "family": "Tree Ensemble", "scaler": "none", "feature_set": "core",
    },
    "HistGradientBoosting": {
        "factory": lambda: HistGradientBoostingRegressor(random_state=RNG_SEED),
        "family": "Tree Ensemble", "scaler": "none", "feature_set": "core",
    },
    "XGBoost": {
        "factory": _xgboost,
        "family": "Boosting", "scaler": "none", "feature_set": "core",
    },
    "LightGBM": {
        "factory": _lightgbm,
        "family": "Boosting", "scaler": "none", "feature_set": "core",
    },
    "CatBoost": {
        "factory": _catboost,
        "family": "Boosting", "scaler": "none", "feature_set": "core",
    },
    "NGBoost": {
        "factory": _ngboost,
        "family": "Boosting (probabilistic)", "scaler": "none", "feature_set": "core",
        "optional": True,
    },
    "EBM": {
        "factory": _ebm,
        "family": "Glass-box (GAM)", "scaler": "none", "feature_set": "core",
        "optional": True,
    },
}
