# Phase 2 — Redundancy Report

Full numbers: [`reports/phase2_analysis_results.json`](phase2_analysis_results.json)
(`redundancy_correlation_matrix`, `vif`, `hierarchical_clusters_at_0.7_abs_corr`).
Figures: [`figures/phase2/engineered_correlation_heatmap.png`](figures/phase2/engineered_correlation_heatmap.png),
[`figures/phase2/feature_dendrogram.png`](figures/phase2/feature_dendrogram.png).

## Method

Spearman correlation matrix across all 24 engineered candidates → hierarchical clustering
on a `1 - |correlation|` distance (average linkage) → cut the dendrogram at distance 0.3
(i.e. merge anything with |correlation| ≥ 0.7) → one representative recommended per cluster.
Variance Inflation Factor (VIF) computed on the standardized 24-feature design matrix as a
second, independent redundancy signal.

## Result: 6 clusters, several near-perfectly collinear

| Cluster | Members | VIF range within cluster | Recommended representative |
|---|---|---|---|
| **Residence** (10 members) | `residence_proxy`, `residence_sq`, `log_residence`, `inv_residence`, `norm_residence`, `severity_index_arrhenius`, `L2_over_F`, `residence_x_temp`, `residence_x_conc`, `avgtemp_x_residence` | 3.6 (`residence_x_conc`) to **∞** (`residence_proxy`, `residence_x_temp`, `avgtemp_x_residence`) | **`residence_proxy`** — simplest, most interpretable, validated in Phase 1/2 |
| **Average temperature** (4 members) | `avg_temp`, `max_temp_approx`, `min_temp_approx`, `arrhenius_avg` | 0.887–0.887 pairwise, all VIF=**∞** in the full 24-feature design | **`avg_temp`** — strongest, most stable standalone correlate (§ stability report); `max_temp_approx` kept as a *secondary* candidate for tree models only (see incremental-value note below) |
| **Thermal gradient** (6 members) | `temp_ratio`, `delta_T`, `norm_delta_T`, `severity_index`, `L_times_deltaT`, `flow_x_deltaT` | 11.5 (`L_times_deltaT`) to 80,619 (`delta_T`, in the full 24-col design where it's reconstructable from several others at once) | **`delta_T`** — simplest, most interpretable; `severity_index` kept as an *additional, justified* feature (survived two independent nested-F tests, §5 incremental value) despite living in this cluster |
| **Absolute gradient** | `abs_delta_T` (singleton — did not merge with the signed-gradient cluster at the 0.7 threshold) | ∞ (in full design) | Kept — captures a genuinely different (V-shaped/magnitude-only) relationship; validated via incremental F-test (p=0.042) |
| **Arrhenius-inlet** | `arrhenius_inlet` (singleton) | 71,354 (full design) | Kept — largest quadratic-shape gain of any feature (§ nonlinearity report) and a highly significant incremental F-test (p=2.1e-5) |
| **Flow transforms** | `inv_flow`, `flow_sq`, `log_flow` | 17–90 | Low priority overall — none of the 3 raw-flow transforms shows a strong standalone correlation with yield (dcor 0.18–0.21, barely above raw `flow_rate_L_min`'s own 0.189); `inv_flow` is the best of a weak bunch if one is needed |

## Reading the VIF numbers correctly

Several features show **VIF = ∞** — this happens when a feature is an *exact* linear
combination of others already in the 24-column design (e.g. `avg_temp` is exactly
reconstructable from `min_temp_approx + delta_T/2`, both of which are also in the design).
**This is expected and mechanical, not a data quality problem** — it is the direct,
quantitative demonstration of exactly the "arbitrary polynomial/interaction explosion" risk
flagged in the phase objective. The practical implication: **never feed all 24 candidates
into a linear model simultaneously** — every cluster above collapses to one meaningfully
independent axis for linear models (though tree-based models can still benefit from having
a couple of "same information, different shape" variants available, since a single axis-
aligned split can't reconstruct a ratio or a difference from two separate columns as
cheaply as having the combined feature pre-computed).

## Net redundancy-based reduction

24 candidates → 6 clusters → **6 representative-or-justified features survive redundancy
screening**: `avg_temp`, `residence_proxy`, `delta_T`, `severity_index`, `abs_delta_T`,
`arrhenius_inlet` (the flow-transform cluster is excluded for weak standalone evidence, not
representativeness). This is the basis for the final recommended feature set — see
[`phase2_feature_engineering_report.md`](phase2_feature_engineering_report.md) §Final Decision.
