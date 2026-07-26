"""Standardized regression metrics -- every model in Phase 4 is scored identically."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "medae": medae, "r2": r2}
