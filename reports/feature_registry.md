# Feature Registry

Single source of truth for every candidate engineered feature across all phases. **No
engineered feature may enter the modeling pipeline unless it first has an entry here.**
Status values: `Proposed` (idea only, untested) → `Implemented` (code exists, not yet
statistically evaluated) → `Validated` (cleared the evidence bar below) / `Rejected`
(evidence argues against it, or superseded by a better version of the same idea).

Validation bar: a feature is `Validated` when it clears **at least one** of (a) a
statistically significant, consistent-sign correlation with the target across ≥2
independent correlation methods, (b) a significant zero-split group effect with
non-negligible effect size, or (c) a significant incremental/partial-F contribution over
correlated alternatives or stated parents. Evidence sources:
[`phase1_eda_findings.md`](phase1_eda_findings.md),
[`phase2_feature_engineering_report.md`](phase2_feature_engineering_report.md),
[`redundancy_report.md`](redundancy_report.md), [`stability_report.md`](stability_report.md).
Code: [`src/features/`](../src/features/).

## Recommended modeling feature set (Final Decision, Phase 2)

Raw 5 + **`avg_temp, residence_proxy, residence_sq, delta_T, severity_index`** (core,
required) + `arrhenius_inlet, abs_delta_T` (optional, validated but no decisive CV benefit
observed yet — available for ablation).

## Full registry

| Feature | Formula | Physical Meaning | Expected Shape | Statistical Support | Complexity | Status |
|---|---|---|---|---|---|---|
| `avg_temp` | `(inlet_temperature_K + jacket_temperature_K) / 2` | Effective/resultant temperature the reacting fluid experiences | **Saturating/threshold (sigmoidal)** — LOWESS shows yield flat ~65-70% below ~410K, sharp decline through ~410-480K, plateaus near 0 above (see figure); quadratic-gain diagnostic (+0.002) is misleading here, a parabola can't fit a one-sided sigmoid | Strongest of all 24 candidates by every method (Pearson −0.637, Spearman −0.711, Kendall −0.517, MI 0.417, dcor 0.670); 100% bootstrap top-8 stability | Low (linear combination) | **Validated — core** |
| `residence_proxy` | `length_m / flow_rate_L_min` | Space/residence time τ proxy (∝ V/Q) | Non-monotonic (inverted-U, τ_opt theory) | Near-null monotonic correlation (Pearson/Spearman/Kendall all ns) but MI=0.339 (top-3 of 24); near-0% bootstrap top-8 by Spearman *but* stable MI (~21% CV) — expected for a non-monotonic feature, not a stability failure | Low (ratio) | **Validated — core** |
| `residence_sq` | `residence_proxy ** 2` | Required to represent τ_opt's interior maximum in a linear model | Quadratic | Same rank-correlation profile as `residence_proxy` (perfectly rank-correlated); incremental F-test vs. `residence_proxy` alone: p=0.111 (borderline); quadratic-shape gain confirmed in shape diagnostics | Low | **Validated — core (paired with residence_proxy, not standalone)** |
| `log_residence` | `log(residence_proxy)` | Compress long-residence right tail | Same rank profile as `residence_proxy` | Incremental F-test vs. `residence_proxy`: p=0.941 (ns) | Low | Rejected — no incremental value over `residence_proxy` |
| `inv_residence` | `flow_rate_L_min / length_m` | Inverse residence (= geometry.flow_per_length, identical formula) | Same rank profile (sign-flipped) | Incremental F-test vs. `residence_proxy`: p=0.502 (ns) | Low | Rejected — redundant with `residence_proxy` (ρ=−1.0) |
| `norm_residence` | z-score of `residence_proxy` | Pure rescaling | Identical to `residence_proxy` | r=1.0 with `residence_proxy`, F-test p=1.0 | Low | Rejected — exact duplicate by construction |
| `max_temp_approx` | `max(inlet_T, jacket_T)` | Upper bound of the temperature envelope (assumes monotonic T(z), a known limitation — ignores possible exothermic hot-spots) | Saturating/threshold, same sigmoidal collapse pattern as `avg_temp` (LOWESS-confirmed; quadratic-gain +0.001 is a misleading diagnostic for this shape) | Spearman −0.694, 100% bootstrap stability; incremental F-test vs. raw temps: p=0.070 (borderline linear), but **largest kNN CV R² gain of any feature (+0.033)** — real nonlinear value a linear model can't exploit | Low | **Validated — secondary (tree models only)** |
| `min_temp_approx` | `min(inlet_T, jacket_T)` | Lower bound of the temperature envelope | Near-monotonic | Spearman −0.488, 98.6% bootstrap stability; F-test vs. raw temps p=0.070 (same as max, mechanically); kNN gain **negative** (−0.026) unlike max | Low | Pending — real correlate but no confirmed model-level benefit distinct from `max_temp_approx`/`avg_temp` |
| `temp_ratio` | `inlet_T / jacket_T` | Relative boundary temperatures | Weak monotonic | Spearman 0.226 (p=0.006); near-perfectly anti-correlated with `norm_delta_T` (ρ=−1.0) and `delta_T` (ρ=−0.998) — pure restatement | Low | Rejected — redundant with `delta_T`/`norm_delta_T` |
| `delta_T` | `jacket_T − inlet_T` | Net thermal driving force / boundary condition | Weak monotonic alone; large effect specifically on the zero-yield split (Phase 1, Cohen's d=0.85) | Spearman −0.240 (p=0.003); F-test vs. raw temps: R² identical (exact linear combination, adds nothing for linear models) | Low | **Validated — core** |
| `abs_delta_T` | `\|delta_T\|` | Captures collapse at *either* extreme (net heating or net cooling), not just net heating | V-shaped | Pearson −0.203 (p=0.013); **incremental F-test vs. `delta_T` alone: p=0.042 (significant)** — confirms the Phase 1 finding that 6/37 zero-yield rows had negative ΔT | Low | **Validated — optional** (no decisive CV gain when benchmarked, §10 of feature engineering report) |
| `norm_delta_T` | `delta_T / avg_temp` | Gradient relative to mean thermal level | Weak monotonic | Incremental F-test vs. `[delta_T, avg_temp]`: **p=0.0037 (significant)** | Low | Pending — statistically real, not yet benchmarked at the model level |
| `arrhenius_inlet` | `exp(-1000 / inlet_temperature_K)` | Arrhenius-form surrogate (fixed, unfitted constant — not literal kinetics) | Strongly nonlinear | Spearman −0.378 (identical to raw `inlet_T`, expected — monotonic transform); **incremental F-test vs. raw `inlet_T`: p=2.1e-5, the strongest incremental result in Phase 2**; largest quadratic-shape gain of all 24 candidates (+0.089) | Low (fixed constant, not fitted) | **Validated — optional** (no decisive CV gain when benchmarked, §10) |
| `arrhenius_avg` | `exp(-1000 / avg_temp)` | Same transform applied to `avg_temp` | Near-linear (quadratic gain +0.0007) | Spearman −0.711 (identical to `avg_temp`); incremental F-test vs. `avg_temp` alone: p=0.109 (ns) | Low | Rejected — functionally redundant with `avg_temp` |
| `severity_index` | `residence_proxy × delta_T` | Damköhler-type severity: duration × net heating, compounded | Captures joint extremes | Two independent nested F-tests both significant (p=0.0067 controlling for avg_temp+delta_T+residence_proxy; p=0.050 controlling for residence_proxy+delta_T only); 0.903 pairwise correlation with `delta_T` alone flagged and resolved via the F-test, not ignored | Low (product) | **Validated — core** |
| `severity_index_arrhenius` | `residence_proxy × arrhenius_avg` | Same severity concept, Arrhenius form instead of linear ΔT | — | Pearson −0.095 (ns), Spearman −0.201 (p=0.014); weaker and less consistent than the `delta_T` version | Low | Rejected (for now) — possible artifact of the arbitrary `ARRHENIUS_C` scale constant; reconsider only with a fitted (not fixed) constant |
| `inv_flow` | `1 / flow_rate_L_min` | Nonlinear flow transform | — | dcor 0.207 (barely above raw flow's 0.189); incremental F-test vs. raw flow: p=0.0037 (significant) but low absolute R² (0.001→0.057) | Low | Pending — statistically real but low practical magnitude; low priority |
| `flow_sq` | `flow_rate_L_min ** 2` | — | — | F-test p=0.192 (ns) | Low | Rejected |
| `log_flow` | `log(flow_rate_L_min)` | — | — | F-test p=0.018 (significant) but low absolute R² | Low | Pending — low priority, same caveat as `inv_flow` |
| `L2_over_F` | `length_m ** 2 / flow_rate_L_min` | Alternative geometric weighting of residence | Redundant with `residence_proxy` (ρ=0.948) | F-test vs. `[length_m, flow_rate_L_min]`: p=0.432 (ns) | Low | Rejected |
| `L_times_deltaT` | `length_m × delta_T` | Cumulative heat-exchange-opportunity proxy | — | F-test vs. `[length_m, delta_T]`: p=0.788 (ns) | Low | Rejected |
| `residence_x_temp` | `residence_proxy × inlet_temperature_K` | Joint residence/temperature driver of conversion | Redundant with `residence_proxy` (ρ=0.993) | F-test p=0.343 (ns) | Low | Rejected |
| `residence_x_conc` | `residence_proxy × concentration_mol_L` | Tests whether concentration matters combined with residence | — | F-test p=0.732 (ns); confirms concentration is null even in interaction | Low | Rejected |
| `avgtemp_x_residence` | `avg_temp × residence_proxy` | Residence weighted by mean temperature | Redundant with `residence_proxy` (ρ=0.995) — numerically dominated by residence_proxy's dynamic range | F-test vs. `[avg_temp, residence_proxy]`: p=0.077 (ns) | Low | Rejected |
| `flow_x_deltaT` | `flow_rate_L_min × delta_T` | Convective-transport framing of thermal gradient (alternative to residence-time framing) | — | F-test vs. `[flow_rate_L_min, delta_T]`: p=0.056 (borderline ns) | Low | Rejected (borderline — could revisit with more data) |
| `inv_inlet_temp` *(Phase 1 feature, 1/T form)* | `1 / inlet_temperature_K` | Earlier, simpler Arrhenius-style transform | — | Superseded by `arrhenius_inlet` (`exp(-1000/T)` form), which passed a much stronger incremental-value test (p=2.1e-5 vs. this feature's untested exponential-form equivalent) | Low | **Rejected — superseded by `arrhenius_inlet`** |

## Rejected-feature summary (why, and reconsideration criteria)

See [`phase2_feature_engineering_report.md`](phase2_feature_engineering_report.md) §11 for
the full discussion. In brief: 17 of 24 Phase 2 candidates were rejected, almost entirely
for one of two reasons — (1) redundancy (perfectly or near-perfectly reconstructable from a
feature already kept, confirmed via correlation matrix + VIF + nested F-tests, not
assumption), or (2) a non-significant incremental F-test against their own stated parents.
No feature was rejected on "it didn't look useful" grounds alone.
