"""Assembles the full candidate feature set from the raw 5 columns.

`ALL_FEATURES` maps feature name -> pure function(df) -> Series. This is the single
place that enumerates every engineered candidate; the feature_registry.md status
column records which of these are actually promoted into a model's feature set.
"""
from __future__ import annotations

import pandas as pd

from . import arrhenius, flow, geometry, interactions, residence, temperature

ALL_FEATURES = {
    # residence family
    "residence_proxy": residence.residence_proxy,
    "residence_sq": residence.residence_sq,
    "log_residence": residence.log_residence,
    "inv_residence": residence.inv_residence,
    "norm_residence": residence.norm_residence,
    # temperature family
    "avg_temp": temperature.avg_temp,
    "max_temp_approx": temperature.max_temp_approx,
    "min_temp_approx": temperature.min_temp_approx,
    "temp_ratio": temperature.temp_ratio,
    "delta_T": temperature.delta_T,
    "abs_delta_T": temperature.abs_delta_T,
    "norm_delta_T": temperature.norm_delta_T,
    # arrhenius-inspired family
    "arrhenius_inlet": arrhenius.arrhenius_inlet,
    "arrhenius_avg": arrhenius.arrhenius_avg,
    "severity_index": arrhenius.severity_index,
    "severity_index_arrhenius": arrhenius.severity_index_arrhenius,
    # flow family
    "inv_flow": flow.inv_flow,
    "flow_sq": flow.flow_sq,
    "log_flow": flow.log_flow,
    # geometry family
    "L2_over_F": geometry.L2_over_F,
    "L_times_deltaT": geometry.L_times_deltaT,
    # interaction family
    "residence_x_temp": interactions.residence_x_temp,
    "residence_x_conc": interactions.residence_x_conc,
    "avgtemp_x_residence": interactions.avgtemp_x_residence,
    "flow_x_deltaT": interactions.flow_x_deltaT,
}

# Which raw feature each engineered feature is most directly derived from --
# used as the "parent" in incremental-value (partial F-test) analysis.
PARENT_OF = {
    "residence_proxy": ["length_m", "flow_rate_L_min"],
    "residence_sq": ["residence_proxy"],
    "log_residence": ["residence_proxy"],
    "inv_residence": ["residence_proxy"],
    "norm_residence": ["residence_proxy"],
    "avg_temp": ["inlet_temperature_K", "jacket_temperature_K"],
    "max_temp_approx": ["inlet_temperature_K", "jacket_temperature_K"],
    "min_temp_approx": ["inlet_temperature_K", "jacket_temperature_K"],
    "temp_ratio": ["inlet_temperature_K", "jacket_temperature_K"],
    "delta_T": ["inlet_temperature_K", "jacket_temperature_K"],
    "abs_delta_T": ["delta_T"],
    "norm_delta_T": ["delta_T", "avg_temp"],
    "arrhenius_inlet": ["inlet_temperature_K"],
    "arrhenius_avg": ["avg_temp"],
    "severity_index": ["residence_proxy", "delta_T"],
    "severity_index_arrhenius": ["residence_proxy", "arrhenius_avg"],
    "inv_flow": ["flow_rate_L_min"],
    "flow_sq": ["flow_rate_L_min"],
    "log_flow": ["flow_rate_L_min"],
    "L2_over_F": ["length_m", "flow_rate_L_min"],
    "L_times_deltaT": ["length_m", "delta_T"],
    "residence_x_temp": ["residence_proxy", "inlet_temperature_K"],
    "residence_x_conc": ["residence_proxy", "concentration_mol_L"],
    "avgtemp_x_residence": ["avg_temp", "residence_proxy"],
    "flow_x_deltaT": ["flow_rate_L_min", "delta_T"],
}


# Phase 2 Final Decision (see reports/phase2_feature_engineering_report.md): the small,
# evidence-validated set that beat both raw-only and raw+all-24-candidates across
# Ridge / RandomForest / CatBoost in repeated-CV benchmarking.
RECOMMENDED_FEATURES_CORE = [
    "avg_temp", "residence_proxy", "residence_sq", "delta_T", "severity_index",
]
# Statistically validated (significant incremental F-tests) but showed no decisive CV RMSE
# gain over the core set alone -- kept available for ablation, not included by default.
RECOMMENDED_FEATURES_OPTIONAL = ["arrhenius_inlet", "abs_delta_T"]


def build_feature_frame(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Returns a DataFrame with one column per requested engineered feature
    (default: all of ALL_FEATURES), computed deterministically from df's raw columns.
    """
    names = feature_names if feature_names is not None else list(ALL_FEATURES.keys())
    out = pd.DataFrame(index=df.index)
    for name in names:
        out[name] = ALL_FEATURES[name](df)
    return out
