"""Distribution transforms and outlier handlers, all leakage-safe (fit-on-train-fold-only
where the transform has any fitted parameter at all).

Validity constraints are enforced explicitly rather than silently skipped:
- log1p, sqrt: require x >= 0 (raises on negative input)
- reciprocal: requires x != 0 (raises on exact zero)
- box_cox: requires strictly positive x (PowerTransformer's own constraint)
- yeo_johnson: valid for any real x (the only transform usable on signed columns
  like delta_T / severity_index)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer


def _log1p(x: np.ndarray) -> np.ndarray:
    if np.any(x < 0):
        raise ValueError("log1p transform requires all values >= 0")
    return np.log1p(x)


def _sqrt(x: np.ndarray) -> np.ndarray:
    if np.any(x < 0):
        raise ValueError("sqrt transform requires all values >= 0")
    return np.sqrt(x)


def _reciprocal(x: np.ndarray) -> np.ndarray:
    if np.any(x == 0):
        raise ValueError("reciprocal transform requires all values != 0")
    return 1.0 / x


def make_transform(name: str):
    """Returns a fresh, unfitted sklearn-compatible transformer for `name`."""
    if name == "none":
        return FunctionTransformer(lambda x: x, feature_names_out="one-to-one")
    if name == "log1p":
        return FunctionTransformer(_log1p, feature_names_out="one-to-one")
    if name == "sqrt":
        return FunctionTransformer(_sqrt, feature_names_out="one-to-one")
    if name == "reciprocal":
        return FunctionTransformer(_reciprocal, feature_names_out="one-to-one")
    if name == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson")
    if name == "box_cox":
        return PowerTransformer(method="box-cox")
    raise ValueError(f"Unknown transform: {name}")


def columns_valid_for_transform(df: pd.DataFrame, columns: list[str], transform_name: str) -> list[str]:
    """Filters `columns` down to those whose values satisfy the transform's domain."""
    valid = []
    for c in columns:
        x = df[c].values
        if transform_name in ("log1p", "sqrt") and np.any(x < 0):
            continue
        if transform_name == "reciprocal" and np.any(x == 0):
            continue
        if transform_name == "box_cox" and np.any(x <= 0):
            continue
        valid.append(c)
    return valid


class SelectiveColumnTransform(BaseEstimator, TransformerMixin):
    """Applies one named transform to a specific subset of columns, passthrough elsewhere.

    Column order in the output always matches the input DataFrame's column order.
    """

    def __init__(self, columns: list[str], transform_name: str):
        self.columns = columns
        self.transform_name = transform_name

    def fit(self, X: pd.DataFrame, y=None):
        self.all_columns_ = list(X.columns)
        transformers = [(c, make_transform(self.transform_name), [c]) for c in self.columns]
        self._ct = ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)
        self._ct.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self._ct.transform(X)
        out_cols = list(self._ct.get_feature_names_out())
        out = pd.DataFrame(arr, columns=out_cols, index=X.index)
        return out[self.all_columns_]  # restore original column order


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips each column to [lower_pct, upper_pct] quantiles fit on the training fold only."""

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X: pd.DataFrame, y=None):
        self.lower_ = X.quantile(self.lower_pct / 100.0)
        self.upper_ = X.quantile(self.upper_pct / 100.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - k*IQR, Q3 + k*IQR] fit on the training fold only."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X: pd.DataFrame, y=None):
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)
