"""Per-model Optuna search spaces. Every bound is justified below -- none are arbitrary.

Phase 4 findings that directly motivate these spaces (see learning_curve_report.md,
residual_analysis_report.md):
  - ExtraTrees reaches near-ZERO training RMSE even at the smallest tested training size
    (n=24) -- it is firmly in a high-variance regime. Its search space is centered on
    REGULARIZATION (shallower trees, larger leaves) rather than more capacity.
  - RandomForest already shows non-zero training RMSE (~8) at defaults -- less overfit than
    ExtraTrees, so its space allows a wider range including "more capacity" directions too.
  - CatBoost's default-hyperparameter learning curve shows the same near-zero-train-RMSE
    high-variance pattern as ExtraTrees -- its space also leans toward regularization
    (lower depth, higher l2_leaf_reg) alongside the standard boosting knobs.
  - GaussianProcess showed noisy, high-variance small-sample training behavior (std=6.8 at
    n=24) and the worst physical-plausibility violation of any competitive model (16%
    outside [0,100]) -- its space specifically searches kernel family and restart count,
    since both directly affect fit stability, not just point RMSE.
"""
from __future__ import annotations

import optuna
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel

RNG_SEED = 42


def suggest_extratrees(trial: optuna.Trial) -> ExtraTreesRegressor:
    # n_estimators: Phase 4 used 300 (a common default-ish choice). More trees only reduces
    # ensemble variance (never increases overfitting) at higher compute cost -- search a
    # range centered on and above the Phase 4 value.
    n_estimators = trial.suggest_int("n_estimators", 100, 800, step=50)
    # max_depth: None (Phase 4 default) grows fully -- exactly what caused the near-zero
    # train RMSE. Search shallower options explicitly to test whether limiting depth trades
    # a little bias for less variance, per the learning-curve diagnosis.
    max_depth = trial.suggest_categorical("max_depth", [3, 5, 8, 12, 20, None])
    # min_samples_leaf=1 (default) lets trees isolate single points -- the direct cause of
    # perfect training-data memorization. Search upward to force averaging over more points
    # per leaf (variance reduction).
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    # max_features: sklearn's current default is 1.0 (all 10 features every split) --
    # restricting this decorrelates trees further (classic RF/ET variance-reduction lever).
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0])
    # bootstrap=False is ExtraTrees' default (whole dataset per tree, only split thresholds
    # randomized). Testing bootstrap=True adds row-sampling as an extra regularizer on top.
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    return ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, max_features=max_features, bootstrap=bootstrap,
        random_state=RNG_SEED,
    )


def suggest_randomforest(trial: optuna.Trial) -> RandomForestRegressor:
    n_estimators = trial.suggest_int("n_estimators", 100, 800, step=50)
    max_depth = trial.suggest_categorical("max_depth", [3, 5, 8, 12, 20, None])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0])
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, max_features=max_features,
        random_state=RNG_SEED,
    )


def suggest_catboost(trial: optuna.Trial):
    from catboost import CatBoostRegressor
    # depth: CatBoost default is 6. Given the overfitting signature, search includes
    # shallower options (4) alongside the default range up to a cap CatBoost recommends
    # rarely exceeding (10) for tabular data this small.
    depth = trial.suggest_int("depth", 4, 10)
    # learning_rate: CatBoost auto-picks based on iterations/dataset size; log-uniform
    # search across the standard practical range.
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    # l2_leaf_reg: default 3.0. Widen substantially upward since the overfitting diagnosis
    # specifically motivates testing much stronger L2 regularization.
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)
    # iterations: capped and paired with early stopping (via eval_set in the objective) so
    # the search doesn't have to separately tune "how many trees" against a fixed number.
    iterations = trial.suggest_int("iterations", 200, 1500, step=100)
    # random_strength: default 1.0, controls randomness in split-score selection (a
    # CatBoost-specific regularizer). Search 0 (off) to well above default.
    random_strength = trial.suggest_float("random_strength", 0.0, 10.0)
    # bagging_temperature: default 1.0, controls the intensity of Bayesian bootstrap
    # weighting -- higher values regularize more aggressively.
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    return CatBoostRegressor(
        depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
        iterations=iterations, random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        verbose=False, random_state=RNG_SEED, allow_writing_files=False,
    )


def suggest_gaussianprocess(trial: optuna.Trial) -> GaussianProcessRegressor:
    # Kernel family: Phase 4 used RBF (infinitely smooth) + WhiteKernel (noise) only.
    # Matern kernels (nu=1.5, 2.5) relax the infinite-smoothness assumption -- plausibly a
    # better fit for a system with the sharp thermal-collapse threshold found in Phase 1/2.
    # RationalQuadratic is a scale-mixture of RBFs -- worth testing given the residence-time
    # non-monotonicity found throughout this project.
    kernel_family = trial.suggest_categorical("kernel_family", ["rbf", "matern1.5", "matern2.5", "rational_quadratic"])
    length_scale = trial.suggest_float("length_scale", 0.1, 10.0, log=True)
    # noise_level (WhiteKernel): default was left at sklearn's default (1.0) in Phase 4.
    # Search explicitly -- an under/over-estimated noise level directly affects both fit
    # quality and the [0,100] plausibility violations Phase 4 flagged as GP's biggest issue.
    noise_level = trial.suggest_float("noise_level", 1e-3, 10.0, log=True)
    # n_restarts_optimizer: Phase 4 used the sklearn default (0 restarts) -- a likely
    # contributor to GP's noisy small-sample training behavior (learning_curve_report.md).
    # Search upward explicitly to test whether more restarts stabilizes the fit.
    n_restarts = trial.suggest_int("n_restarts_optimizer", 0, 10)

    if kernel_family == "rbf":
        base = RBF(length_scale=length_scale)
    elif kernel_family == "matern1.5":
        base = Matern(length_scale=length_scale, nu=1.5)
    elif kernel_family == "matern2.5":
        base = Matern(length_scale=length_scale, nu=2.5)
    else:
        alpha_rq = trial.suggest_float("rq_alpha", 0.1, 10.0, log=True)
        base = RationalQuadratic(length_scale=length_scale, alpha=alpha_rq)

    kernel = base + WhiteKernel(noise_level=noise_level)
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, normalize_y=True, random_state=RNG_SEED,
    )


SUGGEST_FUNCTIONS = {
    "ExtraTrees": suggest_extratrees,
    "RandomForest": suggest_randomforest,
    "CatBoost": suggest_catboost,
    "GaussianProcess": suggest_gaussianprocess,
}
