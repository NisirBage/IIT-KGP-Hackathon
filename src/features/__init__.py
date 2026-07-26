"""Deterministic, reusable physics-informed feature engineering for the reactor dataset.

Every function is a pure transform of the 5 raw columns (`flow_rate_L_min`,
`concentration_mol_L`, `inlet_temperature_K`, `length_m`, `jacket_temperature_K`).
No randomness, no fitted parameters, no leakage from the target.
"""
from .build import (
    ALL_FEATURES,
    RECOMMENDED_FEATURES_CORE,
    RECOMMENDED_FEATURES_OPTIONAL,
    build_feature_frame,
)

__all__ = [
    "build_feature_frame",
    "ALL_FEATURES",
    "RECOMMENDED_FEATURES_CORE",
    "RECOMMENDED_FEATURES_OPTIONAL",
]
