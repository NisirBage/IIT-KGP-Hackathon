"""Arrhenius-inspired surrogate descriptors.

IMPORTANT: we do not know the true activation energies of k1/k2, so these are NOT
literal kinetics. `ARRHENIUS_C` is a fixed, arbitrary scale constant chosen only to
produce meaningful curvature over this dataset's ~350-500K range; it is not fit to
data and carries no claim of physical accuracy. Treat these purely as flexible
nonlinear temperature bases motivated by the exp(-Ea/RT) functional form.
"""
import numpy as np
import pandas as pd

from .residence import residence_proxy
from .temperature import avg_temp

ARRHENIUS_C = 1000.0  # arbitrary fixed scale constant -- NOT a fitted activation energy


def arrhenius_inlet(df: pd.DataFrame) -> pd.Series:
    return np.exp(-ARRHENIUS_C / df["inlet_temperature_K"])


def arrhenius_avg(df: pd.DataFrame) -> pd.Series:
    return np.exp(-ARRHENIUS_C / avg_temp(df))


def severity_index(df: pd.DataFrame) -> pd.Series:
    """Proxy Damkohler-type severity: residence time x net thermal driving force."""
    from .temperature import delta_T
    return residence_proxy(df) * delta_T(df)


def severity_index_arrhenius(df: pd.DataFrame) -> pd.Series:
    """Same severity concept, but using the Arrhenius-transformed average temperature
    instead of the linear delta_T -- tests whether the linear or exponential thermal
    term better captures the interaction."""
    return residence_proxy(df) * arrhenius_avg(df)
