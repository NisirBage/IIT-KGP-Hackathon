# Phase 4 — Residual Analysis Report

Computed on out-of-fold predictions (per-sample mean across the 10 `RepeatedKFold` repeats
— see `baseline_model_report.md` §1 for why this is a clean, genuinely held-out estimate for
every one of the 150 training rows). Full numbers:
[`phase4_analysis_results.json → diagnostics`](phase4_analysis_results.json). Figures:
[`figures/phase4/residual_diagnostics_top6.png`](figures/phase4/residual_diagnostics_top6.png)
(residual-vs-predicted + QQ-plots), [`figures/phase4/residual_distributions_top6.png`](figures/phase4/residual_distributions_top6.png).

## 1. Residual shape (top 6 models by RMSE)

| Model | Mean resid. | Std resid. | Skew | Excess kurtosis | Shapiro p | Heteroscedasticity (Spearman \|resid\| vs. pred) |
|---|---|---|---|---|---|---|
| ExtraTrees | −0.47 | 16.54 | 0.57 | 1.13 | 3.5e-5 | **0.466** (p=1.8e-9) |
| CatBoost | −0.66 | 17.65 | 0.59 | 1.13 | 2.2e-5 | 0.334 (p=2.9e-5) |
| RandomForest | 0.07 | 19.71 | 0.62 | 2.36 | 6.2e-7 | **0.492** (p=1.6e-10) |
| GaussianProcess | −0.52 | 19.90 | 0.54 | 0.97 | 0.023 | 0.163 (p=0.047, weakest of the six) |
| XGBoost | 0.54 | 19.89 | **0.77** | **3.69** | 6.6e-11 | **0.549** (p=3.6e-13) |
| LightGBM | −0.66 | 20.99 | 0.35 | 0.74 | 0.0092 | 0.256 (p=0.0015) |

**Every model's residuals are significantly non-normal** (Shapiro-Wilk p<0.05 for all six) —
expected given the target's own zero-inflated, bounded distribution (Phase 1), not a model
defect per se. **Every model shows positive skew** (0.35–0.77) — large positive residuals
(under-prediction) are more common/extreme than large negative ones, consistent with models
finding the "collapse to near-zero" regime easier to get systematically right than the full
active-yield range.

**Heteroscedasticity is universal and consistent in direction across every model tested**:
`|residual|` correlates positively with the predicted value for all six (strongest for
XGBoost at 0.549, weakest for GaussianProcess at 0.163 but still nominally significant).
**Practical reading: every model is comparatively more accurate near the low/collapsed end
of the yield range and comparatively less precise at high predicted yields** — this is a
shared limitation across model families, not an artifact of any one algorithm, and is a
strong candidate for Phase 8/9 error-region analysis (and a reason the zero-yield-separable
finding from Phase 2 §9 remains worth revisiting for a hurdle-model architecture).

**XGBoost stands out for the wrong reasons here too**: highest skew (0.77) and by far the
highest excess kurtosis (3.69, vs. 0.7–2.4 for the others) — a heavier-tailed, less
well-behaved residual distribution than its leaderboard rank alone would suggest, reinforcing
the instability concern already raised in `model_comparison_report.md`.

## 2. Prediction behavior — physical plausibility ([0, 100] bound)

`overall_yield` is a percentage — any prediction outside [0, 100] is a **concrete, checkable
physical impossibility**, not just an unlikely value. This is the sharpest, least ambiguous
model-quality signal in the whole phase, and several models fail it badly:

| Model | Pred. min | Pred. max | # below 0 | # above 100 | % implausible |
|---|---|---|---|---|---|
| **ExtraTrees** | 0.10 | 91.8 | 0 | 0 | **0.0%** |
| **RandomForest** | 0.02 | 92.6 | 0 | 0 | **0.0%** |
| KNN | 0.01 | 81.7 | 0 | 0 | 0.0% |
| SVR (RBF) | 1.8 | 28.5 | 0 | 0 | 0.0% (meaningless here — see §caveat) |
| CatBoost | −5.0 | 93.5 | 11 | 0 | 7.3% |
| XGBoost | −1.3 | 96.2 | 11 | 0 | 7.3% |
| LinearRegression | −26.2 | 98.1 | 14 | 0 | 9.3% |
| Ridge | −26.2 | 98.7 | 14 | 0 | 9.3% |
| Lasso | −18.3 | 88.0 | 11 | 0 | 7.3% |
| ElasticNet | −7.0 | 78.3 | 3 | 0 | 2.0% |
| LightGBM | −12.2 | 93.6 | 25 | 0 | 16.7% |
| HistGradientBoosting | −10.6 | 95.7 | 27 | 0 | **18.0%** (worst count) |
| **GaussianProcess** | **−18.1** | **112.6** | 22 | 2 | **16.0%** (worst range, only model to also exceed 100) |

**Caveat on SVR's "clean" 0%**: this is not a virtue — SVR's predictions barely span 1.8–28.5,
a symptom of the model failing to learn the target's real range at all (R²=−0.166,
`baseline_model_report.md`), not evidence of well-calibrated bounded behavior.

**This is a genuinely decision-relevant, model-family-correlated finding**: the two bagged
tree ensembles (ExtraTrees, RandomForest) are the *only* competitive models with zero
physically-impossible predictions — an inherent structural property of bagging (a leaf/tree
average can never fall outside the range of training-set target values), not something they
were tuned for. Every boosting method and both continuous-domain models
(GaussianProcess, all 4 linear models) can and do extrapolate outside [0, 100] at default
settings. This was weighed directly in the Final Recommendation
(`model_comparison_report.md` §6) — GaussianProcess's 16% implausible rate is its single
biggest liability, and CatBoost's more modest 7.3% still needs a post-hoc clipping step
before any prediction leaves the pipeline.

## 3. Prediction stability across folds

`mean_fold_to_fold_std` = average, across all 150 samples, of the standard deviation of that
sample's prediction across the 10 independent `RepeatedKFold` repeats (i.e. how much a given
row's prediction moves depending on which random fold split it happened to land in).

| Model | Mean fold-to-fold std | Max fold-to-fold std |
|---|---|---|
| ElasticNet | 2.04 | 7.53 |
| ExtraTrees | **2.76** | 11.03 |
| Lasso | 2.74 | 12.82 |
| RandomForest | 3.61 | 13.87 |
| SVR | 3.63 | 6.58 |
| KNN | 3.84 | 10.25 |
| CatBoost | 3.97 | 11.11 |
| LinearRegression | 4.42 | 12.46 |
| GaussianProcess | 5.41 | 15.50 |
| HistGradientBoosting | 6.08 | 15.49 |
| LightGBM | 6.26 | 16.50 |
| **XGBoost** | **6.97** | **24.07** |

Among the top-tier models, **ExtraTrees is both the most accurate and the most stable**
(lowest mean fold-to-fold std of the non-linear models). **XGBoost is the least stable model
in the entire benchmark** by a wide margin (mean std 6.97, worst-case single-sample std
24.07 — for at least one training row, XGBoost's prediction swings by 24 percentage points
depending on which fold split it landed in). Combined with its elevated `rmse_std` (4.01,
`baseline_model_report.md`) and heavy-tailed residuals (§1), this is a consistent,
multi-angle case for treating XGBoost cautiously despite a respectable point RMSE — exactly
the situation the phase brief anticipated ("reject unstable models even if their average
RMSE is slightly better").

## Summary

No model is residual-diagnostically "clean" — universal non-normality and heteroscedasticity
are shared, structural properties of this problem (bounded, zero-inflated target), not fixable
by model choice alone. What *does* differentiate models here is (a) physical plausibility —
only bagged trees respect the [0,100] bound inherently — and (b) stability — ExtraTrees is
both accurate and stable, XGBoost is neither exceptional nor safe despite a mid-table RMSE.
Both properties fed directly into the Final Recommendation in `model_comparison_report.md`.
