# Phase 3 — Leakage Validation Report

This report exists to answer one question with evidence, not assertion: **does building
preprocessing inside a `sklearn.Pipeline` and running it through `cross_val_score` actually
prevent validation-fold information from leaking into preprocessing, and does it matter
numerically on this dataset?**

Code under test: [`src/preprocessing/validation.py`](../src/preprocessing/validation.py),
[`src/preprocessing/feature_selector.py`](../src/preprocessing/feature_selector.py),
[`src/preprocessing/pipelines.py`](../src/preprocessing/pipelines.py).

## Check 1: is preprocessing actually refit per fold, or silently shared across folds?

`check_fold_specific_fitting` fits `FeatureSetSelector` + `StandardScaler` on the **training
split only** of each of 5 `KFold` folds, and records the fitted `StandardScaler.mean_` for
the first column (`flow_rate_L_min`) each time. If preprocessing were accidentally fit once
on the full dataset (a common, easy-to-introduce leakage bug — e.g. calling
`scaler.fit(X)` before `train_test_split` or before the CV loop), every fold would report the
**identical** fitted mean. If it is genuinely refit per fold (each fold excluding a different
20% validation slice), the fitted means will differ slightly from fold to fold.

**Result — fitted `mean_` across the 5 folds:**

```
Fold 0: 41.207
Fold 1: 41.672
Fold 2: 39.111
Fold 3: 40.202
Fold 4: 40.142
```

All five values are different. This confirms the pipeline is doing genuine per-fold
refitting, not accidentally sharing a single global fit — the mechanism this phase's core
requirement ("no statistic computed from the full training dataset may influence a
validation fold") depends on.

## Check 2: does the leakage actually bias the reported RMSE, and by how much?

`leaky_vs_correct_cv` runs the *same* model (`Ridge`, same `KFold(5, seed=42)` splits) two
ways:

- **Leaky**: fit `FeatureSetSelector` + `StandardScaler` **once on the full 150-row
  dataset**, transform everything, then run `cross_val_score` on the already-transformed
  matrix. Every fold's "training" scaling statistics have already seen that fold's own
  validation rows.
- **Correct**: the same model wrapped in the full `build_pipeline(...)`, run through
  `cross_val_score` directly on raw `X` — scikit-learn refits every pipeline step on each
  fold's training split internally.

**Single-seed result:**

| | RMSE |
|---|---|
| Leaky | 30.5438 |
| Correct | 30.6551 |

The leaky estimate is lower (more optimistic) by 0.11 — in the expected direction (using
validation-fold information to fit scaling makes the reported error look smaller than it
truly is), but a single split isn't enough to know if that's systematic or noise.

**Repeated over 30 independent `KFold` reshufflings (seeds 0–29):**

| | Value |
|---|---|
| Mean(leaky − correct) | **−0.288** |
| Seeds where leaky < correct (optimistic) | **29 / 30** |
| Range of (leaky − correct) | −1.660 to +0.023 |

**This is the decisive result**: in 29 of 30 independent train/validation splits, the leaky
procedure reports a lower (more optimistic) RMSE than the correct, pipeline-based procedure.
This is not noise from one unlucky split — it is a systematic, repeatable bias in the
direction leakage theory predicts. The magnitude here (~0.29 RMSE units, against a baseline
RMSE of ~30) is modest — expected for `StandardScaler` specifically, since a mean/std fit on
150 rows vs. ~120 rows (a fold's training split) doesn't differ dramatically at this sample
size — but **the direction and consistency, not the magnitude, is the point**: a more
sensitive preprocessing step (e.g. `QuantileTransformer`, or feature construction that pools
information more aggressively) would very plausibly show a much larger bias on a dataset
this small, which is exactly why every preprocessing step in this project's pipeline is
built to be fit-per-fold by construction, not by discipline.

## Check 3: engineered features that need fitted statistics

Phase 2 flagged `norm_residence` (a z-scored version of `residence_proxy`) as a specific
leakage risk, since its Phase-2 implementation fit mean/std from whatever DataFrame it was
given. `FeatureSetSelector.fit()` (§ code) now stores `residence_proxy`'s mean/std from the
fold's training data only, and `transform()` reuses those fitted values on any data —
verified working end-to-end in the Phase 3 module smoke test (the `all_candidates` feature
set, which includes `norm_residence`, runs through `cross_val_score` without error and
without needing any special-casing at the call site). Every other engineered feature in
`src/features/` is a pure row-wise function of the raw columns (no cross-row statistics), so
no other feature carries this risk.

## Conclusion

All three checks pass: preprocessing is refit per fold (Check 1), the fit-per-fold discipline
has a measurable, systematically-signed effect on the reported metric (Check 2), and the one
identified stateful engineered feature is now leakage-safe by construction (Check 3). The
`build_pipeline(...)` factory is the only sanctioned way to combine feature construction,
scaling, transforms, and a model in this project from here on — no step should ever be fit
outside of it.
