# Phase 4 — Learning Curve Report

Generated for the top-tier cluster identified in `model_comparison_report.md` §3
(ExtraTrees, CatBoost, RandomForest, GaussianProcess) via `sklearn.model_selection.learning_curve`,
8 training-set sizes from 20% to 100% of the training data,
`RepeatedKFold(5, 3)` (a lighter CV budget than the main benchmark's 5×10, for runtime —
this analysis is about the *shape* of the curve, not a precise point estimate). Full data:
[`phase4_learning_curves.json`](phase4_learning_curves.json). Figure:
[`figures/phase4/learning_curves_top4.png`](figures/phase4/learning_curves_top4.png).

## Results at full training size (n=120 per fold)

| Model | Train RMSE | Validation RMSE | Gap |
|---|---|---|---|
| ExtraTrees | ≈0.00 (2.3×10⁻¹³) | 16.68 | 16.68 |
| CatBoost | 0.50 | 17.71 | 17.21 |
| RandomForest | 7.80 | 19.78 | 11.98 |
| GaussianProcess | 8.18 | 20.68 | 12.50 |

## The central finding: none of the four validation curves has plateaued

Every model's validation RMSE is **still decreasing** from the second-to-last to the last
training-size point:

- ExtraTrees: 17.58 → 16.68 (still falling)
- CatBoost: 18.18 → 17.71 (still falling)
- RandomForest: 19.88 → 19.78 (smallest final-step improvement of the four — closest to a
  plateau, but not flat)
- GaussianProcess: 21.10 → 20.68 (still falling)

**Answer to the phase's explicit question ("is the model data-limited or
algorithm-limited?"): all four top models show clear evidence of being at least partially
data-limited.** None has exhausted the benefit of the 150 training rows available — more
data would very plausibly improve every one of them further. This is a directly useful,
concrete answer for the hackathon pitch's "how would this scale to more plant data"
question (explicitly named as a judged criterion in the problem statement).

## Bias/variance decomposition, per model

- **ExtraTrees and CatBoost are firmly in the high-variance regime**: both reach
  near-zero training error even at the *smallest* tested training size (24 samples) —
  ExtraTrees' train RMSE is ~10⁻¹³ at every single training size tested, i.e. it
  memorizes the training fold regardless of how small it is. The large, slowly-narrowing
  train-validation gap (26.6→16.7 as n grows for ExtraTrees) is a textbook high-variance
  signature. **These two models would benefit from both more data and from Phase 5
  regularization tuning that deliberately trades a little bias for less variance**
  (`max_depth`, `min_samples_leaf` for ExtraTrees; `depth`, `l2_leaf_reg` for CatBoost).
- **RandomForest shows a smaller variance gap than ExtraTrees despite an architecturally
  similar bagging setup**: its training RMSE stays around 8–10 rather than collapsing to
  zero, even though both use `n_estimators=300` and unrestricted default depth. This is a
  genuinely interesting, slightly counter-intuitive empirical result — plausibly related to
  how `RandomForestRegressor`'s bootstrap-with-replacement row sampling interacts with its
  best-split (CART) search differently than `ExtraTreesRegressor`'s random-threshold splits,
  but this project did not dig into the sklearn internals far enough to confirm a precise
  mechanism, and that explanation is offered as a plausible hypothesis, not a verified one.
  Practically: RandomForest is already closer to its achievable bias floor than ExtraTrees,
  so its Phase 5 tuning ceiling from *regularization* is probably smaller — most of its
  remaining gain is more likely to come from more data than from hyperparameter search.
- **GaussianProcess shows unstable small-sample behavior**: training RMSE is noisy and
  non-monotonic across sizes (7.17 → 8.75 → 6.93 → 3.84 → 7.59 → 7.10 → 7.98 → 8.18), with a
  very large standard deviation at the smallest size tested (std=6.83 at n=24, settling to
  ~2.7 by n≥78). This is a distinct stability concern from the ones raised for XGBoost
  elsewhere (`residual_analysis_report.md` §3) but points the same direction: **GP's Phase 5
  tuning should prioritize kernel/prior choices that behave sensibly with limited data**,
  not just squeezing out marginal RMSE.

## Implication for Phase 5 search budget

Given none of the four curves has plateaued, Phase 5's hyperparameter search should not be
read as "the final word" on any of these models' ceilings even after tuning — some of the
remaining gap to a hypothetically-perfect model is a data-size ceiling this project cannot
lift (only 150 labeled rows exist). Tuning should focus on **closing the train-validation gap
for ExtraTrees/CatBoost** (variance reduction) and on **kernel/prior robustness for
GaussianProcess**, while treating any RMSE improvement from more data (not available here) as
a separate, unaddressable axis worth naming explicitly in the final pitch's "how this scales"
discussion.
