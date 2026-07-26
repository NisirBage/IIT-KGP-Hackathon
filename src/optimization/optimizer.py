"""Runs (or resumes) an Optuna study for one model, with SQLite-backed storage so every
completed trial is checkpointed to disk immediately -- re-running this function with the
same model_name resumes the existing study rather than starting over, which is what makes
this framework resilient to this environment's background-execution reliability issues
(see baseline_model_report.md / preprocessing_report.md for prior incidents).
"""
from __future__ import annotations

from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .callbacks import make_progress_logger
from .configs import N_TRIALS, RNG_SEED
from .objective import make_objective

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = PROJECT_ROOT / "artifacts" / "optuna_studies"
STUDY_DIR.mkdir(parents=True, exist_ok=True)


def run_study(model_name: str, X, y, n_trials: int | None = None) -> optuna.Study:
    storage = f"sqlite:///{STUDY_DIR / (model_name + '.db')}"
    sampler = TPESampler(seed=RNG_SEED)
    pruner = MedianPruner(n_warmup_steps=2)  # let each trial complete >=2 folds before pruning eligibility

    study = optuna.create_study(
        study_name=model_name, storage=storage, direction="minimize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    already_done = len([t for t in study.trials if t.state in (
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    target = n_trials if n_trials is not None else N_TRIALS[model_name]
    remaining = max(0, target - already_done)
    print(f"[{model_name}] {already_done} trials already checkpointed, running {remaining} more (target={target})")

    if remaining > 0:
        objective = make_objective(model_name, X, y)
        study.optimize(objective, n_trials=remaining, callbacks=[make_progress_logger(model_name)])

    return study


def load_study(model_name: str) -> optuna.Study:
    storage = f"sqlite:///{STUDY_DIR / (model_name + '.db')}"
    return optuna.load_study(study_name=model_name, storage=storage)
