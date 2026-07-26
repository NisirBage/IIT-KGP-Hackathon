"""Builds the Optuna objective function for a given model: fits the trial's suggested
hyperparameters through the model's Phase-3-recommended preprocessing pipeline, evaluates
with the model-specific objective CV budget (configs.OBJECTIVE_CV_BUDGET), and reports
fold-level intermediate values so Optuna's pruner can stop clearly-bad trials early
(applies uniformly to all 4 models). CatBoost additionally gets native early stopping on its
boosting iterations within each fit (a distinct mechanism from trial-level pruning).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.pipelines import build_pipeline  # noqa: E402
from models.registry import MODEL_REGISTRY  # noqa: E402

from .configs import objective_cv
from .search_spaces import SUGGEST_FUNCTIONS


def make_objective(model_name: str, X, y):
    suggest_fn = SUGGEST_FUNCTIONS[model_name]
    reg_entry = MODEL_REGISTRY[model_name]
    cv = objective_cv(model_name)

    def objective(trial: optuna.Trial) -> float:
        base_model = suggest_fn(trial)
        fold_rmses = []
        for i, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipe = build_pipeline(clone(base_model),
                                   feature_set=reg_entry["feature_set"], scaler=reg_entry["scaler"])

            if model_name == "CatBoost":
                # Native early stopping: carve a small validation slice out of THIS fold's
                # training data (never touches the fold's held-out test rows).
                X_fit, X_es, y_fit, y_es = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
                # Build the pipeline's preprocessing on X_fit, transform X_es the same way,
                # then fit CatBoost directly with eval_set for early stopping.
                pre = pipe[:-1]
                pre.fit(X_fit, y_fit)
                X_fit_t = pre.transform(X_fit)
                X_es_t = pre.transform(X_es)
                cb_model = pipe.named_steps["model"]
                cb_model.set_params(early_stopping_rounds=50)
                cb_model.fit(X_fit_t, y_fit, eval_set=(X_es_t, y_es), verbose=False)
                X_te_t = pre.transform(X_te)
                pred = cb_model.predict(X_te_t)
            else:
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_te)

            rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
            fold_rmses.append(rmse)

            trial.report(float(np.mean(fold_rmses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

    return objective
