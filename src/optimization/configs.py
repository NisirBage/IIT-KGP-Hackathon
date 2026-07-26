"""Shared Optuna optimization configuration.

Objective CV budget: RepeatedKFold(5,3) = 15 folds per trial for the tree models
(ExtraTrees, RandomForest -- cheap per-fit) and RepeatedKFold(5,2) = 10 folds for the more
expensive models (CatBoost, GaussianProcess) -- a deliberate search-efficiency trade-off
(Core Principle 3), justified quantitatively in validation_strategy_report.md. This is NOT
the protocol used to report final tuned-vs-baseline numbers -- every model selected by
Optuna is re-evaluated with the full RepeatedKFold(5,10) (identical to the Phase 4 baseline
protocol) before any tuned-vs-baseline claim is made, so no decision in this project is ever
based on a single CV estimate (Core Principle 1).
"""
from sklearn.model_selection import RepeatedKFold

RNG_SEED = 42

OBJECTIVE_CV_BUDGET = {
    "ExtraTrees": {"n_splits": 5, "n_repeats": 3},
    "RandomForest": {"n_splits": 5, "n_repeats": 3},
    "CatBoost": {"n_splits": 5, "n_repeats": 2},
    "GaussianProcess": {"n_splits": 5, "n_repeats": 2},
}

# Target counts, scaled down from the phase's "100-200 trials" guidance after a 5-trial
# smoke test measured ~13s/trial for RandomForest (~65s/5 trials) under this objective CV
# budget -- 150 trials would be ~30+ min per tree model alone. Reduced to keep total search
# time tractable within this session while still giving Optuna's TPE sampler a reasonable
# sample (60-80 trials is well within the range where TPE shows clear convergence behavior
# for a 5-6 dimensional space); see optimization_diagnostics_report.md for the convergence
# check that confirms whether more trials would likely have helped. SQLite checkpointing
# (optimizer.py) means these can be resumed and extended later without losing progress.
N_TRIALS = {
    "ExtraTrees": 80,
    "RandomForest": 80,
    "CatBoost": 60,
    "GaussianProcess": 25,  # "fewer, more expensive trials" per phase spec
}


def objective_cv(model_name: str):
    budget = OBJECTIVE_CV_BUDGET[model_name]
    return RepeatedKFold(n_splits=budget["n_splits"], n_repeats=budget["n_repeats"], random_state=RNG_SEED)


# Final re-validation protocol -- identical to the Phase 4 baseline benchmark, so tuned vs.
# baseline numbers are directly, fairly comparable.
FINAL_VALIDATION_CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG_SEED)
