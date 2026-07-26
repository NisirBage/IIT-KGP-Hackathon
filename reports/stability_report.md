# Phase 2 — Stability Report

Method: 500 bootstrap resamples of the 150 training rows (with replacement). For each
resample, recompute Spearman correlation of every engineered feature vs. `overall_yield`
(all 500 resamples) and Mutual Information (every 10th resample, 50 total — MI estimation
is the slow step). Track: (a) mean/std of each statistic across resamples, (b) how often
each feature lands in the top-8 by |Spearman| across all 500 rankings. Full numbers:
[`phase2_analysis_results.json → stability`](phase2_analysis_results.json).

## Headline result: the strongest features are also the most stable

| Feature | Spearman mean ± std | % of bootstraps in top-8 by \|Spearman\| | MI mean ± std |
|---|---|---|---|
| **avg_temp** | −0.709 ± 0.041 | **100%** | 0.792 ± 0.141 |
| **arrhenius_avg** | −0.709 ± 0.041 | **100%** | 0.794 ± 0.140 |
| **max_temp_approx** | −0.691 ± 0.044 | **100%** | 0.794 ± 0.155 |
| min_temp_approx | −0.483 ± 0.065 | 98.6% | 0.638 ± 0.144 |
| arrhenius_inlet | −0.371 ± 0.068 | 84.8% | 0.651 ± 0.158 |
| severity_index | −0.281 ± 0.088 | 75.6% | 0.560 ± 0.137 |
| L_times_deltaT | −0.259 ± 0.082 | 70.6% | 0.533 ± 0.145 |
| delta_T | −0.241 ± 0.077 | 52.8% | 0.540 ± 0.134 |
| residence_proxy (and its monotonic-transform siblings) | −0.084 ± 0.088 | **0.2%** | 0.691 ± 0.146 |

The temperature-level cluster (`avg_temp`/`arrhenius_avg`/`max_temp_approx`) is not just the
strongest correlate — it is essentially **noise-free across resampling**: a coefficient of
variation of ~6% on the Spearman estimate and never once falling out of the top-8 across
500 independent resamples. This is about as strong a stability result as this dataset can
produce and substantially de-risks building the final model around this feature.

## The `residence_proxy` case: a stability metric mismatch, not an instability finding

`residence_proxy` (and its perfectly rank-correlated siblings `residence_sq`,
`log_residence`, `inv_residence`, `norm_residence`) essentially never appears in the
top-8-by-|Spearman| ranking (0–0.2% of resamples) — at first glance this looks like an
*unstable, unreliable* feature. **It is not** — this is the same non-monotonic signature
from Phase 1/2's correlation battery, now showing up correctly in the bootstrap: a feature
whose relationship with the target is a genuine inverted-U cannot score well on a
rank-correlation-based stability metric in any single resample, bootstrap or not, because
Spearman is structurally blind to non-monotonic dependence. The relevant stability check for
this feature is its **Mutual Information stability**, which is in fact reasonably good
(mean 0.691, std 0.146 → ~21% coefficient of variation, in the same range as most other
features in this table) — i.e., the *elevated MI finding itself* is stable across
resamples, even though its *rank* by a metric that can't see its shape correctly is not.
**Lesson applied to feature selection: never screen candidate features using a single
stability metric that structurally cannot detect their expected relationship shape.**

## Practical implication for Phase 3+

No feature in the final recommended set (`avg_temp`, `residence_proxy`, `delta_T`,
`severity_index`, `abs_delta_T`, `arrhenius_inlet`) shows evidence of a fragile, sample-luck
correlation — every one of them is either in the "extremely stable by Spearman" group or the
"stable-by-MI, structurally invisible-to-Spearman" group, not a "looks strong once, evaporates
on resampling" case. That specific failure mode — which is the main risk this bootstrap
analysis was designed to catch on a 150-row dataset — was not observed for any promoted
feature.
