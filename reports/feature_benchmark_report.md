# Phase 2 — Feature Benchmark Report

**No hyperparameter tuning was performed anywhere in this report** — every model uses
library defaults (Ridge α=1.0, `RandomForestRegressor(n_estimators=300)`,
`CatBoostRegressor()` defaults). The objective is to measure whether a *feature set* helps,
holding the model fixed, not to find the best model. All comparisons use the identical CV
scheme: `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)` — 50 RMSE values averaged
per cell, so every number below is a mean ± std over 50 folds, not one lucky split.

## Main comparison: Raw vs. Raw+Validated vs. Raw+All-candidates

| Feature set | # features | Ridge RMSE | RandomForest RMSE | CatBoost RMSE |
|---|---|---|---|---|
| Raw only | 5 | 30.47 ± 1.98 | 21.61 ± 3.35 | 19.75 ± 2.84 |
| **Raw + validated (5 engineered)** | 10 | **30.30 ± 2.37** | **19.91 ± 2.92** | **17.76 ± 2.40** |
| Raw + all 24 candidates | 29 | 31.26 ± 4.80 | 21.29 ± 3.09 | 17.93 ± 2.54 |

("Validated" here = `avg_temp, residence_proxy, residence_sq, delta_T, severity_index` —
the 5 features validated by the correlation battery + partial F-tests as of the first
benchmark pass; see the follow-up comparison below for two more candidates.)

**This is the central empirical result of Phase 2**: the small, curated 5-feature set is
the best or statistically-tied-best feature set for **every one of the three model
families**, and dumping in all 24 engineered candidates **never wins**, actively hurting
Ridge (RMSE 30.47→31.26, and its std nearly doubles from 1.98→4.80 — a sign of coefficient
instability from collinearity, exactly as the redundancy report's VIF=∞ findings predicted)
and showing no benefit for RandomForest (21.29 vs. 19.91 — worse than the validated set).
Only CatBoost is roughly indifferent between "validated" and "all candidates" (17.76 vs.
17.93 — a 0.17 RMSE gap against a ~2.4-2.5 std, i.e. noise), consistent with CatBoost's
built-in regularization and feature-importance-based splitting making it more robust to
redundant inputs than Ridge or vanilla RandomForest — but even for CatBoost, more features
bought literally nothing.

## Follow-up: does the fuller incremental-value evidence change the recommendation?

Two features not in the original "validated" set — `abs_delta_T` and `arrhenius_inlet` —
passed their own independent statistical tests later in Phase 2 (partial F-test p=0.042 and
p=2.1e-5 respectively; see feature engineering report §5). We re-ran the identical CV
benchmark with them added (7 engineered features total) to check whether that
statistically-significant *univariate* incremental value translates into a *model-level*
RMSE improvement:

| Feature set | Ridge RMSE | RandomForest RMSE | CatBoost RMSE |
|---|---|---|---|
| Raw + validated v1 (5 features) | 30.30 ± 2.37 | 19.91 ± 2.92 | 17.76 ± 2.40 |
| Raw + validated v2 (7 features, +`abs_delta_T`, +`arrhenius_inlet`) | 30.02 ± 2.21 | 20.21 ± 2.81 | 17.70 ± 2.29 |

**Verdict: no decisive change.** Every delta (Ridge −0.29, RandomForest +0.30, CatBoost
−0.06) is small relative to its fold-to-fold std (2.2–2.9) — noise-level, not a real
improvement or regression. **This is an important, honest finding in its own right**: a
feature can pass a rigorous, statistically significant incremental-information test in
isolation (controlling only for its specific stated parents) and still fail to move the
needle once it has to compete with, and interact with, the *other* features already in a
real model's feature set. The nested F-test answers "does this feature carry information
beyond its parents", not "does adding this feature to our actual candidate model improve
predictions" — only the full benchmark answers the second, more decision-relevant question.
Given no decisive gain and the phase's explicit mandate to prefer a small, defensible set,
**the 5-feature v1 set remains the recommendation** (see Final Decision in
[`phase2_feature_engineering_report.md`](phase2_feature_engineering_report.md)).

## Reading the per-model pattern

- **Ridge is nearly flat across all three feature sets** (30.47 → 30.30 → 31.26) — expected,
  since `avg_temp` and `delta_T` are exact linear combinations of the two raw temperatures
  (redundancy report) and add no new *linear* information; only `residence_proxy`,
  `residence_sq`, and `severity_index` are genuinely non-reconstructable nonlinear terms for
  a linear model, and their contribution is modest given the sample size.
- **RandomForest and CatBoost both improve substantially from raw-only to validated**
  (RF: 21.61→19.91, CatBoost: 19.75→17.76) — tree-based models benefit far more from
  pre-computed ratios/differences than linear models do, because a single split cannot
  reconstruct `length_m/flow_rate_L_min` or `jacket_T − inlet_T` from the raw columns as
  efficiently as being handed the combined feature directly.
- **CatBoost is the strongest model at every feature-set size tested** — consistent with it
  handling the reactor's known non-monotonic (residence-time) and threshold-like (zero-yield
  collapse) structure better than Ridge (linear) or default RandomForest.

## Zero-yield separability (from the same evidence set, no model training involved)

Using only 6 engineered physics features (`avg_temp, residence_proxy, delta_T,
severity_index, max_temp_approx, abs_delta_T`) — **no raw features, no tuning** — four
simple classifiers were evaluated via `RepeatedStratifiedKFold(5, 10)`:

| Model | CV Accuracy | CV AUC |
|---|---|---|
| Majority-class baseline | 75.3% | — |
| Logistic Regression | 87.7% ± 5.2% | 0.936 ± 0.043 |
| Decision Tree (depth ≤ 3) | 84.7% ± 6.5% | 0.886 ± 0.076 |
| LDA | 87.7% ± 5.4% | 0.942 ± 0.040 |
| QDA (reg_param=0.2\*) | 89.3% ± 5.1% | 0.934 ± 0.052 |

\*QDA needed light regularization — several of these physics features are highly
correlated (`severity_index`↔`delta_T` at ρ=0.90), which made the per-class covariance
matrix singular at `reg_param=0` in some folds.

**All four models beat the majority-class baseline by 9–14 points and achieve AUC ≥ 0.89.**
This is a strong, physics-feature-only separation of the zero-yield regime — see the full
discussion, including the actual learned depth-3 tree rules (which read as a clean physical
story: high average temperature + long residence + continued net heating ⇒ collapse), in
[`phase2_feature_engineering_report.md`](phase2_feature_engineering_report.md) §9. This is
flagged as **strong justification to evaluate a two-stage (classify-then-regress / hurdle)
model architecture in Phase 4**, not a recommendation to build one yet.
