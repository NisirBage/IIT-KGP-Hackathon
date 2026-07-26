"""Runs (or resumes) the Optuna study for one model, named on the command line.
Usage: python run_phase5_optimize.py <ModelName> [n_trials_this_call]
SQLite storage means this can be called repeatedly / interrupted and resumed freely.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import TARGET_COL, load_train  # noqa: E402
from optimization import run_study  # noqa: E402
from optimization.configs import N_TRIALS  # noqa: E402

model_name = sys.argv[1]
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else None

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values

if chunk is not None:
    # Run exactly `chunk` more trials this call (resuming from wherever the study left off),
    # regardless of the model's total N_TRIALS target -- lets a long study be run in
    # several short, foreground-safe calls.
    from optimization.optimizer import STUDY_DIR
    import optuna
    storage = f"sqlite:///{STUDY_DIR / (model_name + '.db')}"
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optimization.configs import RNG_SEED
    from optimization.objective import make_objective
    from optimization.callbacks import make_progress_logger

    study = optuna.create_study(
        study_name=model_name, storage=storage, direction="minimize",
        sampler=TPESampler(seed=RNG_SEED), pruner=MedianPruner(n_warmup_steps=2),
        load_if_exists=True,
    )
    objective = make_objective(model_name, X, y)
    study.optimize(objective, n_trials=chunk, callbacks=[make_progress_logger(model_name)])
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    print(f"[{model_name}] now at {done}/{N_TRIALS[model_name]} trials total")
else:
    study = run_study(model_name, X, y)
    print(f"[{model_name}] study complete. Best value: {study.best_value:.4f}")
    print(f"[{model_name}] Best params: {study.best_params}")
