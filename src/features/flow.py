"""Flow-rate-derived features (standalone nonlinear transforms of flow_rate_L_min)."""
import numpy as np
import pandas as pd


def inv_flow(df: pd.DataFrame) -> pd.Series:
    return 1.0 / df["flow_rate_L_min"]


def flow_sq(df: pd.DataFrame) -> pd.Series:
    return df["flow_rate_L_min"] ** 2


def log_flow(df: pd.DataFrame) -> pd.Series:
    return np.log(df["flow_rate_L_min"])
