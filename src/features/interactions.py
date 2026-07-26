"""Physically-motivated interaction terms only -- no arbitrary polynomial explosion.

Each interaction here is included because Phase 0/1 evidence gave a specific reason
to expect it (see reports/phase1_eda_findings.md and phase2_feature_engineering_report.md).
"""
import pandas as pd

from .residence import residence_proxy
from .temperature import avg_temp, delta_T


def residence_x_temp(df: pd.DataFrame) -> pd.Series:
    """Residence time x inlet temperature -- both jointly set how much A converts."""
    return residence_proxy(df) * df["inlet_temperature_K"]


def residence_x_conc(df: pd.DataFrame) -> pd.Series:
    """Residence time x inlet concentration -- tests whether concentration matters
    only in combination with residence time (Phase 1 found it null standalone)."""
    return residence_proxy(df) * df["concentration_mol_L"]


def avgtemp_x_residence(df: pd.DataFrame) -> pd.Series:
    """Residence time x average (not inlet-only) temperature."""
    return avg_temp(df) * residence_proxy(df)


def flow_x_deltaT(df: pd.DataFrame) -> pd.Series:
    """Flow rate x net thermal gradient -- tests a convective-heat-transport framing
    (mass flow carrying a thermal gradient) as an alternative to the residence-time
    framing used by severity_index."""
    return df["flow_rate_L_min"] * delta_T(df)
