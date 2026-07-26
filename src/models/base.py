"""Shared types and constants for the Phase 4 model benchmark framework."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelResult:
    """Container for one model's full repeated-CV benchmark output."""
    model: str
    family: str
    fold_rows: list[dict]
    oof_predictions: np.ndarray  # shape (n_repeats, n_samples)
    fit_time_mean: float
    predict_time_mean: float
    summary: dict = field(default_factory=dict)
