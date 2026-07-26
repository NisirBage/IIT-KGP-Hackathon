# Phase 3 — Preprocessing Report

Code: [`src/preprocessing/`](../src/preprocessing/) (reusable module — `feature_selector.py`,
`transformers.py`, `scalers.py`, `pipelines.py`, `validation.py`, `config.py`). Every result
below comes from an actual `RepeatedKFold` cross-validated pipeline run, not a single split.
**Note on execution**: the original single monolithic analysis script was killed partway
through by the environment when run in the background (twice, at different points — no
error, no traceback, process simply disappeared). All results were still obtained, split
across a foreground run and two lighter foreground follow-up scripts
(`run_phase3_analysis.py`, `_part2.py`, `_part3.py`); the CV budget for the two follow-up
scripts was reduced from 5×10 to 5×5 repeats specifically to keep runtime low after the
background failures — noted wherever it applies below.

---

## 1. Feature distribution audit (core feature set: 5 raw + 5 engineered)

| Feature | Skewness | Excess kurtosis | Min | Max | IQR outliers | Sign |
|---|---|---|---|---|---|---|
| flow_rate_L_min | 0.14 | −1.26 | 5.41 | 79.02 | 0 | + |
| concentration_mol_L | −0.14 | −1.22 | 0.52 | 3.97 | 0 | + |
| inlet_temperature_K | −0.07 | −1.26 | 351.6 | 498.6 | 0 | + |
| length_m | −0.07 | −1.22 | 2.26 | 24.99 | 0 | + |
| jacket_temperature_K | 0.27 | −1.11 | 354.0 | 548.0 | 0 | + |
| avg_temp | 0.22 | −0.72 | 363.5 | 516.2 | 0 | + |
| **residence_proxy** | **2.77** | **11.08** | 0.036 | 4.49 | **15 (10%)** | + |
| **residence_sq** | **7.06** | **62.15** | 0.001 | 20.19 | **21 (14%)** | + |
| delta_T | 0.05 | −0.84 | −128.1 | 176.6 | 0 | ± |
| severity_index | 0.64 | 4.21 | −165.3 | 209.2 | 18 (12%) | ± |

Zero missing values anywhere (consistent with Phase 1's dataset passport).

**This audit directly determines what's justified below, rather than applying transforms
by default**: the 5 raw features and `avg_temp` are all mild, platykurtic, outlier-free —
**no distributional treatment is justified for these**. `residence_proxy` (and its square)
is the one genuinely heavy-tailed, high-outlier-count feature — a real candidate for
transformation. `delta_T` is already essentially symmetric (skew 0.05) — **audit-level
evidence against transforming it before even running the benchmark**. `severity_index` sits
in between — moderate skew and kurtosis, worth testing but not a strong prior either way.

## 2. Scaling strategy (per model family, `RepeatedKFold(5,10)`, core feature set)

| Scaler | Ridge | SVR | KNN | GaussianProcess |
|---|---|---|---|---|
| none | 30.27 ± 2.42 | 42.67 ± 5.37 | **24.72 ± 1.79** | 38.41 ± 2.60 |
| standard | 30.56 ± 2.57 | 40.78 ± 5.15 | 26.54 ± 2.56 | 22.22 ± 2.57 |
| robust | 30.58 ± 2.58 | 43.03 ± 5.67 | 28.92 ± 3.36 | 23.95 ± 2.89 |
| minmax | 30.07 ± 2.02 | **40.55 ± 5.01** | 25.92 ± 2.22 | 21.77 ± 2.37 |
| quantile_normal | 32.02 ± 3.35 | 41.87 ± 5.42 | 26.84 ± 2.03 | 24.00 ± 2.79 |
| power_yeojohnson | **29.57 ± 2.27** | 40.92 ± 5.24 | 25.04 ± 2.15 | **20.55 ± 2.45** |
| power_boxcox\* | 30.42 ± 1.96 | 41.16 ± 5.28 | 26.75 ± 2.22 | 21.25 ± 2.42 |

\*Box-Cox requires strictly positive input across every column it's applied to; the core
set includes signed columns (`delta_T`, `severity_index`), so Box-Cox was benchmarked on
the `raw_only` feature set instead (all 5 raw features are strictly positive) — **not a
like-for-like comparison with the other rows**, included for completeness per the phase
objective, not as a fair ranking entry.

**Findings, by model family:**
- **Ridge**: every scaler is within ~1 std of every other (29.6–32.0 vs std ~2–3.3) —
  **no scaler choice is decisive here**, consistent with Phase 2's finding that Ridge barely
  responds to feature engineering on this dataset. `power_yeojohnson` is the best point
  estimate and is retained as the recommendation on that basis, not because the difference
  is statistically compelling.
- **SVR**: **scaling clearly matters** — unscaled is the worst by a wide margin (42.67),
  `minmax` and `standard` are the best, roughly tied (40.55 vs 40.78). Expected: SVR's
  default RBF-like kernel is scale-dependent.
- **KNN**: **the unscaled feature set is the best result of the entire table** (24.72),
  meaningfully ahead of every scaler tried, with `robust` actively the worst (28.92). This
  contradicts the usual "always scale for distance-based models" default — see the
  discussion below.
- **GaussianProcess**: **scaling is essential** (unscaled: 38.41, more than 15 RMSE points
  worse than the best scaled option) — expected, since the default RBF+White kernel assumes
  roughly unit-scale features. `power_yeojohnson` is the clear best (20.55).

**On the KNN result specifically**: this is counter-intuitive enough to flag explicitly
rather than silently override with "best practice." A plausible explanation: `residence_proxy`
and `residence_sq` are heavily right-skewed with real outliers (§1) — standardizing them to
unit variance gives those noisy, heavy-tailed dimensions equal weight in the Euclidean
distance metric alongside the cleaner, well-behaved temperature features, which can hurt a
distance-based method more than leaving the natural (already roughly comparable — flow ~5-80,
temperatures ~350-550, residence ~0-4.5) raw scales alone. This is a hypothesis consistent
with the evidence, not independently re-verified against a KNN run with only the
well-behaved subset of features — flagged for a Phase 4 ablation if KNN is pursued further.

**Tree models — confirmatory check** (`RandomForest`, `CatBoost`, 3 scalers): RMSE identical
to 3 decimal places regardless of scaler (RandomForest: 19.926/19.925/19.927 for
none/standard/robust; CatBoost: 17.987/17.986/17.987) — **exactly confirms the theoretical
expectation that tree-based split-finding is invariant to monotonic per-feature rescaling**,
empirically, not just by assumption.

## 3. Distribution transforms (Ridge, core+standard-scaled baseline RMSE = 30.56)

| Column | Transform | Skew before → after | Ridge RMSE |
|---|---|---|---|
| residence_proxy | log1p | 2.77 → 1.40 | 30.38 |
| residence_proxy | sqrt | 2.77 → 1.22 | 30.24 |
| residence_proxy | reciprocal | 2.77 → 2.30 | 29.99 |
| residence_proxy | **yeo_johnson** | 2.77 → **0.23** | 30.26 |
| residence_proxy | box_cox | 2.77 → 0.00 | 30.25 |
| severity_index | reciprocal | 0.64 → **9.43 (worse!)** | 32.01 (worse) |
| severity_index | yeo_johnson | 0.64 → 0.11 | 30.74 (~flat) |
| delta_T | reciprocal | 0.05 → −1.14 (worse) | 30.84 (worse) |
| delta_T | yeo_johnson | 0.05 → −0.05 (~unchanged) | 30.48 (~flat) |

`log1p`/`sqrt`/`box_cox` were correctly skipped for `delta_T`/`severity_index` by the
transformer's built-in domain check (both columns contain negative values) — not omitted by
oversight; this is the validity-enforcement described in `transformers.py`.

**Findings:**
- All 5 valid transforms on `residence_proxy` give a small, consistent RMSE improvement
  (29.99–30.38 vs 30.56 baseline) — modest but directionally consistent with the audit's
  finding that this is the one genuinely heavy-tailed feature. `yeo_johnson` and `box_cox`
  do the best job of actually fixing the skew (→0.23 and →0.00), so either is defensible;
  `yeo_johnson` is preferred going forward because it is valid on signed columns too
  (uniform treatment across the whole feature set, see §4 below).
- **`reciprocal` is actively harmful on both signed columns** (`severity_index` skew
  0.64→9.43, `delta_T` skew 0.05→−1.14) — a textbook reciprocal-transform pathology: both
  columns cross (or nearly cross) zero, and `1/x` blows up near zero, manufacturing extreme
  values rather than removing them. **Explicitly do not use `reciprocal` on any column that
  isn't bounded away from zero.**
- **`delta_T` gains nothing from any transform** — expected given its audit-stage skew was
  already only 0.05; this is the rare case where the pre-transform audit and the
  post-transform benchmark agree perfectly, and no further test was needed to reach the
  "don't transform this" conclusion.
- **`severity_index`**: Yeo-Johnson fixes the skew nicely (0.64→0.11) but does not improve
  (mildly worsens) CV RMSE (30.74 vs 30.56 baseline) — a case, like Phase 2's
  `abs_delta_T`/`arrhenius_inlet` finding, where a real distributional improvement doesn't
  translate to a model-level benefit. **Recommendation: do not transform `severity_index`.**

## 4. A cross-check that changed the final scaler recommendation's implementation detail

The main pipeline benchmark (§ `pipeline_benchmark_report.md`) tested Yeo-Johnson applied
*only* to the 3 flagged skew columns (`SelectiveColumnTransform`) plus a separate
`StandardScaler` on everything — that combination scored 30.54 for Ridge, **meaningfully
worse than applying `power_yeojohnson` as a single global scaler across all 10 core columns
(29.57, §2)**. Both use the same underlying transform; the difference is scope (3 columns
vs. all 10). **Conclusion: for Ridge, apply Yeo-Johnson globally as the scaler, not as a
selective per-column pre-step** — simpler pipeline, better result. This is the final
recommendation's actual shape (see `pipeline_benchmark_report.md`'s Final Recommendation
table): no separate "skew transform" stage survives for any model family once the global
Yeo-Johnson-as-scaler option is on the table.

## 5. Outlier strategy (`RepeatedKFold(5,5)`\* — reduced budget, see note above)

| Model | none | winsorize_1_99 | clip_iqr |
|---|---|---|---|
| Ridge (standard-scaled) | 30.61 ± 2.60 | **30.31 ± 2.66** | 30.66 ± 2.11 |
| RandomForest | 19.96 ± 3.22 | 19.98 ± 3.28 | 20.27 ± 3.38 |
| CatBoost | 17.94 ± 2.83 | 18.10 ± 2.82 | **17.90 ± 2.77** |

No outlier strategy produces a change larger than its own fold-to-fold noise for any model
— **no decisive evidence that outlier handling matters here**, for either linear or tree
models. This matches the Phase 1 finding that the IQR-flagged "outliers" in `residence_proxy`
etc. are not anomalies but the natural tail of a skewed-but-legitimate operating-condition
distribution (Phase 0/1 confirmed no impossible values anywhere in the raw data).
**Per the explicit instruction not to alter/remove physically meaningful operating regimes
without justification, and given no measured benefit, the default recommendation is `none`**
for every model family. `winsorize_1_99` is retained as an available, low-risk *optional*
setting for Ridge specifically (smallest, most consistent of the marginal effects observed),
never as a default.

## Next step

See [`pipeline_benchmark_report.md`](pipeline_benchmark_report.md) for the main
raw/scaled/transformed/scaled+transformed comparison and the final per-model-family
recommendation table, and [`leakage_validation_report.md`](leakage_validation_report.md)
for the explicit, empirical leakage-prevention demonstration.
