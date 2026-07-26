"""Residence-time (space-time) proxy features.

Physical basis: for a fixed reactor cross-section, space time tau ~ V/Q = (A*L)/Q,
so tau is proportional to length_m / flow_rate_L_min up to the (unknown, assumed
constant) cross-sectional area A. Series-reaction theory (A->B->C) predicts an
*interior maximum* of yield vs tau -- see reports/phase0_problem_understanding.md.
"""
import numpy as np
import pandas as pd


def residence_proxy(df: pd.DataFrame) -> pd.Series:
    return df["length_m"] / df["flow_rate_L_min"]


def residence_sq(df: pd.DataFrame) -> pd.Series:
    """Quadratic term -- required to represent an interior maximum with a linear model."""
    return residence_proxy(df) ** 2


def log_residence(df: pd.DataFrame) -> pd.Series:
    """Log transform -- compresses the right tail (rare very-long residence rows)."""
    return np.log(residence_proxy(df))


def inv_residence(df: pd.DataFrame) -> pd.Series:
    """Inverse residence = flow_rate_L_min / length_m (identical formula to
    geometry.flow_per_length -- kept as two names because Phase 2 spec lists both
    the 'residence' and 'geometry' families separately; see redundancy report."""
    return df["flow_rate_L_min"] / df["length_m"]


def norm_residence(df: pd.DataFrame) -> pd.Series:
    """Z-scored residence proxy. This is a pure rescaling of residence_proxy and
    is expected to carry zero incremental information beyond it (r=1.0) -- included
    to demonstrate the redundancy-detection pipeline actually catches trivial cases."""
    r = residence_proxy(df)
    return (r - r.mean()) / r.std(ddof=1)
