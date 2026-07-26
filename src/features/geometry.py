"""Reactor-geometry-derived features.

Note: L/F is identical to residence.residence_proxy and flow_per_length is
identical to residence.inv_residence -- both are implemented once in `residence.py`
and referenced here to avoid duplicate columns; see redundancy report for the
explicit collinearity check that confirms this equivalence numerically.
"""
import pandas as pd

from .temperature import delta_T


def L2_over_F(df: pd.DataFrame) -> pd.Series:
    """length^2 / flow -- alternative geometric weighting of residence time."""
    return (df["length_m"] ** 2) / df["flow_rate_L_min"]


def L_times_deltaT(df: pd.DataFrame) -> pd.Series:
    """Total thermal driving force integrated over reactor length (a crude proxy for
    cumulative heat-exchange opportunity)."""
    return df["length_m"] * delta_T(df)
