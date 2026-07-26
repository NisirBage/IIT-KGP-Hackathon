# Preprocessing Registry

Single source of truth for every preprocessing decision, analogous to
[`feature_registry.md`](feature_registry.md) for features. No preprocessing choice should
be changed in Phase 4+ without updating this table and citing new evidence. Full evidence:
[`preprocessing_report.md`](preprocessing_report.md), [`pipeline_benchmark_report.md`](pipeline_benchmark_report.md),
[`leakage_validation_report.md`](leakage_validation_report.md).

## Feature set

| Decision | Value | Evidence | Status |
|---|---|---|---|
| Default feature set for all models | `core` (5 raw + `avg_temp, residence_proxy, residence_sq, delta_T, severity_index`) | Phase 2 Final Decision, re-confirmed as the feature set used throughout every Phase 3 benchmark | **Adopted** |

## Scaling, per model family

| Model | Scaler | Evidence | Status |
|---|---|---|---|
| Ridge | `power_yeojohnson` (global, all 10 core columns) | Best point estimate among 7 scalers tested (29.57 vs. next-best minmax 30.07); beat a selective-3-column YJ + standard-scaler combination (30.54) by ~1 RMSE point | **Adopted** (weak — every Ridge scaler choice is within ~1 std of every other) |
| SVR | `standard` | Scaling essential (unscaled 42.67 vs. scaled ~40.5-40.8); standard and minmax statistically tied | **Adopted** |
| KNN | `none` | Unscaled (24.72) beat every one of 6 scalers tested, including standard (26.54) and the runner-up power_yeojohnson (25.04) | **Adopted** (counter-intuitive; hypothesis in preprocessing_report.md §2 not independently re-verified) |
| GaussianProcess | `power_yeojohnson` | Scaling essential (unscaled 38.41 vs. best scaled 20.55); YJ clearly best of 6 scalers tested | **Adopted** |
| RandomForest | `none` | RMSE identical to 3 decimals across none/standard/robust (19.925-19.927) | **Adopted** (confirmed scale-invariant) |
| CatBoost | `none` | RMSE identical to 3 decimals across none/standard/robust (17.986-17.987) | **Adopted** (confirmed scale-invariant) |
| `power_boxcox` (any model) | — | Requires strictly positive input; core feature set has signed columns (`delta_T`, `severity_index`) — **not usable on the core set as a global scaler** | **Rejected** for the core set; only tested on `raw_only` for reference |
| `quantile_normal` (any model) | — | Consistently the worst or near-worst scaler for every model tested (Ridge 32.02, SVR 41.87, KNN 26.84, GP 24.00) | **Rejected** |
| `robust` (any model) | — | Never the best scaler for any model; actively worst for KNN (28.92) | **Rejected** as a default; no evidence it's needed over standard/YJ anywhere |

## Distribution transforms

| Column | Transform | Evidence | Status |
|---|---|---|---|
| `residence_proxy` | `yeo_johnson` (as part of the global Ridge scaler, not a separate stage) | Skew 2.77→0.23 after transform; Ridge RMSE improves modestly (30.56→29.57 as global scaler) | **Adopted** (folded into scaler choice, not a standalone pipeline stage) |
| `residence_proxy` | `reciprocal` | Numerically best single-column RMSE (29.99) but worst skew fix (2.77→2.30) of the 5 tested; less interpretable and not needed once global YJ scaling is adopted | **Rejected** (redundant given global YJ scaler choice) |
| `severity_index` | `reciprocal` | **Actively harmful**: skew 0.64→9.43 (much worse), RMSE 32.01 (worse than baseline 30.56) | **Rejected — do not use reciprocal on signed, near-zero-crossing columns** |
| `severity_index` | `yeo_johnson` | Fixes skew (0.64→0.11) but RMSE 30.74, mildly worse than 30.56 baseline | **Rejected** — real distributional improvement, no model-level benefit |
| `delta_T` | any transform | Audit-stage skew already only 0.05 (near-symmetric); both tested transforms (reciprocal, yeo_johnson) show no RMSE benefit and reciprocal actively worsens skew (→−1.14) | **Rejected** — this column needs no distributional treatment |
| `log1p`, `sqrt`, `box_cox` on `delta_T`/`severity_index` | — | Both columns contain negative values; these transforms require non-negative (or strictly positive) input | **Not applicable** (correctly excluded by the transformer's domain validity check, not by oversight) |

## Outlier handling

| Model | Strategy | Evidence | Status |
|---|---|---|---|
| All models | `none` (default) | No outlier strategy tested produced a change larger than fold-to-fold noise for any model (Ridge: 30.61→30.31 winsorize, within ~2.6 std; RandomForest/CatBoost: flat) | **Adopted as default** — also required by the explicit instruction not to alter physically meaningful operating regimes without justification (Phase 1 confirmed no impossible values) |
| Ridge | `winsorize_1_99` | Smallest, most consistent of the marginal effects tested (30.31 vs. 30.61 none) — never combined/re-tested with the `power_yeojohnson` scaler | **Optional** — available, not default, not decisively validated in combination with the adopted scaler |
| Any model | `clip_iqr` | No consistent benefit for any model; slightly worse than `none` for Ridge and RandomForest | **Rejected** as a default |

## Leakage-prevention mechanisms

| Component | Mechanism | Validation | Status |
|---|---|---|---|
| `FeatureSetSelector` | Proper `fit`/`transform` Estimator; `norm_residence`'s mean/std fit only on the data passed to `.fit()` | Fold-specific fitted-mean check (leakage_validation_report.md Check 1) confirms per-fold refitting | **Validated** |
| All scalers/transforms | Assembled via `build_pipeline(...)` as `sklearn.Pipeline` steps, never fit standalone before CV | Leaky-vs-correct CV comparison, 30 seeds: 29/30 show the leaky version is optimistically biased (mean −0.29 RMSE) | **Validated** |
| Pipeline serialization | `joblib.dump`/`load` round-trip | Predictions identical after reload (`max_abs_diff=0.0`) | **Validated** |
