"""Scaler registry -- factory functions (not shared instances) so every pipeline build
gets a fresh, unfitted transformer."""
from __future__ import annotations

from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def make_scaler(name: str):
    if name == "none":
        return None
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "quantile_normal":
        # n_quantiles capped below n_samples to avoid a sklearn warning at n=150 with CV folds
        return QuantileTransformer(output_distribution="normal", n_quantiles=100, random_state=42)
    if name == "power_yeojohnson":
        return PowerTransformer(method="yeo-johnson")
    if name == "power_boxcox":
        return PowerTransformer(method="box-cox")
    raise ValueError(f"Unknown scaler: {name}")
