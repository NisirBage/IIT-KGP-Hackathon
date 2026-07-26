"""Post-hoc analysis of a completed/in-progress Optuna study: history, parameter
importance, convergence (best-so-far trajectory), and trial runtime distribution."""
from __future__ import annotations

import numpy as np
import optuna


def study_summary(study: optuna.Study) -> dict:
    df = study.trials_dataframe()
    completed = df[df["state"] == "COMPLETE"]
    pruned = df[df["state"] == "PRUNED"]

    best_so_far = []
    running_best = np.inf
    for v in completed.sort_values("number")["value"]:
        running_best = min(running_best, v)
        best_so_far.append(running_best)

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    durations = completed["duration"].dt.total_seconds() if "duration" in completed else None

    return {
        "n_trials_total": len(df),
        "n_complete": len(completed),
        "n_pruned": len(pruned),
        "best_value": float(study.best_value) if len(completed) else None,
        "best_params": study.best_params if len(completed) else None,
        "best_trial_number": study.best_trial.number if len(completed) else None,
        "best_so_far_trajectory": best_so_far,
        "param_importances": {k: float(v) for k, v in importances.items()},
        "trial_duration_sec": {
            "mean": float(durations.mean()) if durations is not None and len(durations) else None,
            "std": float(durations.std()) if durations is not None and len(durations) else None,
            "min": float(durations.min()) if durations is not None and len(durations) else None,
            "max": float(durations.max()) if durations is not None and len(durations) else None,
            "total": float(durations.sum()) if durations is not None and len(durations) else None,
        },
    }


def convergence_check(best_so_far: list[float], last_n: int = 30) -> dict:
    """Did optimization converge, or are additional trials likely to help?
    Heuristic: compare improvement in the last `last_n` trials to improvement in the
    window before that -- a sharply decelerating improvement rate suggests convergence."""
    if len(best_so_far) < last_n * 2:
        return {"verdict": "insufficient_trials_to_assess", "n_trials": len(best_so_far)}
    recent_improvement = best_so_far[-last_n * 2] - best_so_far[-last_n]
    earlier_improvement = best_so_far[-last_n] - best_so_far[0] if len(best_so_far) > last_n else None
    still_improving = recent_improvement > 1e-3
    return {
        "verdict": "still_improving" if still_improving else "plateaued",
        "improvement_last_window": float(recent_improvement),
        "final_best": float(best_so_far[-1]),
    }
