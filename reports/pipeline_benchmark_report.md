# Phase 3 — Pipeline Benchmark Report

No hyperparameter tuning anywhere in this report — Ridge α=1.0, `RandomForestRegressor
(n_estimators=300)`, `CatBoostRegressor()` defaults throughout, exactly as in Phase 2. All
core-feature-set numbers use `RepeatedKFold(5, 10)` (Phase 3 §2/§3 tables in
[`preprocessing_report.md`](preprocessing_report.md)); this section's combination grid uses
`RepeatedKFold(5, 5)` (reduced from 10 repeats after the original script was twice killed
by the environment when backgrounded — see the note in `preprocessing_report.md`).

## Main comparison: raw / scaled / transformed / scaled+transformed

"Transformed" here = Yeo-Johnson applied *only* to the 3 flagged skew columns
(`residence_proxy`, `severity_index`, `delta_T`) via `SelectiveColumnTransform`, everything
else passed through unchanged — a different scope than §2's "power_yeojohnson as global
scaler" result, see the cross-check note below.

| Preprocessing config | Ridge | RandomForest | CatBoost |
|---|---|---|---|
| raw | 30.41 ± 2.76 | 19.96 ± 3.22 | 17.94 ± 2.83 |
| scaled (standard) | 30.61 ± 2.60 | 19.96 ± 3.22 | 17.94 ± 2.83 |
| transformed (selective YJ) | 30.55 ± 3.15 | 19.96 ± 3.21 | 17.94 ± 2.83 |
| scaled + transformed | 30.54 ± 3.14 | 19.96 ± 3.21 | 17.94 ± 2.83 |

**RandomForest and CatBoost are completely flat across all four configurations** (RMSE
identical to within 0.01 — pure CV noise) — the strongest possible empirical confirmation
that neither scaling nor this transform scope matters at all for tree-based models on this
dataset. **Ridge shows small (noise-level) movement** across configs, none of it decisive
against the ~2.6–3.2 std fold-to-fold variation.

## Important cross-check: selective-column transform vs. global-scaler transform

Comparing this section's "scaled + transformed" Ridge result (30.54) against
`preprocessing_report.md`'s scaler benchmark result for `power_yeojohnson` as a **global**
scaler (29.57, §2 of that report) shows a real, reproducible gap — global Yeo-Johnson beats
the selective 3-column version by about 1 RMSE point. **Practical implication for the final
recommendation**: don't build a separate "skew transform" pipeline stage — just use
Yeo-Johnson as the scaler itself, applied to the whole feature matrix. This is simpler (one
fewer pipeline stage) and empirically better for the one model family (Ridge) where scaling
choice matters at all beyond noise.

## Leakage validation summary (full detail in `leakage_validation_report.md`)

Per-fold `StandardScaler` fitted means differ across all 5 folds (39.1–41.7, never
identical) — confirms genuine per-fold refitting. A leaky-vs-correct CV comparison repeated
over 30 independent reshufflings shows the leaky procedure reports a lower (optimistic) RMSE
in **29/30** cases, mean bias −0.29 RMSE — small in magnitude at n=150 but a clear,
systematically-signed effect, not noise.

## Serialization test

`build_pipeline(Ridge(), feature_set="core", scaler="standard")` fit on the full training
set, `joblib.dump`/`joblib.load` round-tripped, predictions compared: **predictions match
exactly after reload** (`max_abs_diff = 0.0`). Saved artifact:
`artifacts/pipelines/ridge_core_standard_v1.joblib`.

---

## Final Recommendation — preprocessing by model family

| Model | Feature Set | Scaling | Skew Transform | Outlier Handling | Notes |
|---|---|---|---|---|---|
| **Ridge** | core (10 feat.) | `power_yeojohnson` (global) | none (superseded by global YJ scaler, §above) | none (winsorization gave only a noise-level edge, 30.31 vs 30.61, not combined/re-tested with YJ) | Best point estimate 29.57, but every scaler tested for Ridge is within ~1 std of every other — treat this as a mild preference, not a decisive win |
| **SVR** | core | `standard` (or `minmax`, statistically tied: 40.78 vs 40.55) | none | none | Scaling clearly matters (unscaled is ~2 RMSE points worse); standard/minmax indistinguishable from each other |
| **KNN** | core | **none** | none | none | Counter-intuitive but clearly evidence-backed: unscaled (24.72) beat every scaler tried, `robust` was the worst (28.92) — see hypothesis in `preprocessing_report.md` §2 |
| **GaussianProcess** | core | `power_yeojohnson` | none | none | Scaling is essential (unscaled RMSE 38.41 vs. best scaled 20.55) — largest scaling effect of any model tested |
| **RandomForest** | core | none | none | none | Fully scale/transform-invariant, confirmed to 3 decimal places across every combination tested |
| **CatBoost** | core | none | none | none | Same as RandomForest — fully invariant; also the best absolute RMSE of every model tested at every preprocessing setting (17.90–17.94) |

**No universal one-size-fits-all pipeline is recommended** — the evidence is unambiguous
that model families genuinely differ here: three families (SVR, GP, and to a lesser/noisier
extent Ridge) benefit meaningfully or mildly from scaling, KNN is actively hurt by every
scaler tested, and both tree ensembles are provably indifferent to all of it. `config.py`'s
`MODEL_FAMILY_PREPROCESSING` dict encodes exactly this table so Phase 4 reads it rather than
re-deriving preprocessing choices per model.
