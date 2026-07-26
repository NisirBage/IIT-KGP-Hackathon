"""The Phase 6 recommended ensemble: a linear blend of ExtraTrees + CatBoost + RandomForest
base pipelines (each already includes feature engineering + scaling per Phase 3/5), combined
via fixed, pre-fit linear coefficients (fit out-of-fold across the full Phase 4/5/6 OOF
history -- see reports/phase6_final_3model_blend.json), clipped to the physically valid
[0,100] yield range.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LinearBlendEnsemble(BaseEstimator, RegressorMixin):
    """Combines fitted base pipelines with fixed linear weights + clipping.

    `base_pipelines` must be a dict of {name: fitted sklearn Pipeline}. `coefficients` and
    `intercept` are fixed (not re-fit here) -- they were determined out-of-fold in Phase 6
    (see reports/phase6_final_3model_blend.json) and are treated as part of the ensemble's
    definition, not something `.fit()` learns from whatever data it's given.
    """

    def __init__(self, base_pipelines: dict, coefficients: dict, intercept: float,
                 clip_low: float = 0.0, clip_high: float = 100.0):
        self.base_pipelines = base_pipelines
        self.coefficients = coefficients
        self.intercept = intercept
        self.clip_low = clip_low
        self.clip_high = clip_high

    def fit(self, X, y=None):
        for pipe in self.base_pipelines.values():
            pipe.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        pred = np.full(len(X), self.intercept)
        for name, pipe in self.base_pipelines.items():
            pred = pred + self.coefficients[name] * pipe.predict(X)
        return np.clip(pred, self.clip_low, self.clip_high)
