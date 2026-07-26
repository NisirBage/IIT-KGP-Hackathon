"""Optuna callbacks. SQLite storage (configured in optimizer.py) already gives per-trial
checkpointing for free -- these callbacks are for logging/visibility during a run."""
from __future__ import annotations

import time

import optuna


def make_progress_logger(model_name: str):
    start = time.perf_counter()

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        elapsed = time.perf_counter() - start
        state = trial.state.name
        print(
            f"[{model_name}] trial {trial.number:4d}  state={state:8s}  "
            f"value={trial.value if trial.value is not None else float('nan'):8.3f}  "
            f"best={study.best_value:8.3f}  elapsed={elapsed:6.1f}s",
            flush=True,
        )

    return callback
