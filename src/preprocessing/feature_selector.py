"""Leakage-safe, sklearn-compatible feature construction + selection.

This is the FIRST step of every pipeline built in this project. It is a proper
fit/transform Estimator so that when it lives inside a sklearn Pipeline, cross_val_score
(or any CV splitter) refits it independently on every training fold -- no engineered
feature is ever built using statistics from a validation fold.

Almost every engineered feature in src/features/ is a pure row-wise function of the raw
columns (no cross-row statistics), so "fitting" is a no-op for them. The one exception
flagged in Phase 2 is `norm_residence` (a z-score of residence_proxy) -- its mean/std MUST
be fit on the training fold only. This transformer is what actually closes that leakage
loophole: fit() stores residence_proxy's mean/std from whatever data it is given (which,
inside a Pipeline+CV, is always the training fold only), and transform() re-uses those
fitted values on any data (train or validation/test).
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from features import ALL_FEATURES, build_feature_frame
from features.residence import residence_proxy
from utils.io import FEATURE_COLS


class FeatureSetSelector(BaseEstimator, TransformerMixin):
    """Builds engineered features and returns exactly the requested columns.

    Parameters
    ----------
    feature_names : list[str]
        Column names to return, drawn from FEATURE_COLS (raw) union ALL_FEATURES.keys()
        (engineered). Order is preserved.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y=None):
        self._residence_mean_ = None
        self._residence_std_ = None
        if "norm_residence" in self.feature_names:
            r = residence_proxy(X)
            self._residence_mean_ = float(r.mean())
            self._residence_std_ = float(r.std(ddof=1))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        engineered_needed = [f for f in self.feature_names if f in ALL_FEATURES and f != "norm_residence"]
        engineered = build_feature_frame(X, engineered_needed) if engineered_needed else pd.DataFrame(index=X.index)

        out = pd.DataFrame(index=X.index)
        for name in self.feature_names:
            if name in FEATURE_COLS:
                out[name] = X[name].values
            elif name == "norm_residence":
                if self._residence_mean_ is None:
                    raise RuntimeError("FeatureSetSelector.fit() must be called before transform() when norm_residence is requested.")
                r = residence_proxy(X)
                out[name] = (r - self._residence_mean_) / self._residence_std_
            else:
                out[name] = engineered[name].values
        return out

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names)
