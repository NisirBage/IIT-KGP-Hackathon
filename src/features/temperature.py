"""Temperature-derived features.

Physical basis: local temperature along the reactor is not observed directly -- only
the two boundary values (inlet_temperature_K, jacket_temperature_K) are given. These
features are simple, interpretable summaries of that boundary pair, motivated by the
energy-balance coupling described in reports/phase0_problem_understanding.md.
"""
import numpy as np
import pandas as pd


def avg_temp(df: pd.DataFrame) -> pd.Series:
    return (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0


def max_temp_approx(df: pd.DataFrame) -> pd.Series:
    """Upper bound of the temperature envelope under the simplifying assumption that
    local T moves monotonically between the two boundary values (ignores any
    exothermic-reaction hot-spot that could exceed both -- a known limitation)."""
    return np.maximum(df["inlet_temperature_K"], df["jacket_temperature_K"])


def min_temp_approx(df: pd.DataFrame) -> pd.Series:
    return np.minimum(df["inlet_temperature_K"], df["jacket_temperature_K"])


def temp_ratio(df: pd.DataFrame) -> pd.Series:
    return df["inlet_temperature_K"] / df["jacket_temperature_K"]


def delta_T(df: pd.DataFrame) -> pd.Series:
    return df["jacket_temperature_K"] - df["inlet_temperature_K"]


def abs_delta_T(df: pd.DataFrame) -> pd.Series:
    return delta_T(df).abs()


def norm_delta_T(df: pd.DataFrame) -> pd.Series:
    """Delta T relative to the mean thermal level -- a dimensionless version of the
    net heating gradient."""
    return delta_T(df) / avg_temp(df)
