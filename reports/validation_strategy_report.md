# Phase 5 — Validation Strategy Report

Before any tuning: which validation protocol should be trusted, and for what? All four
protocols below were run on **ExtraTrees at its fixed Phase 4 default hyperparameters** (this
step evaluates the *protocol*, not the model) on the identical training data. Raw numbers:
[`phase5_validation_strategy_results.json`](phase5_validation_strategy_results.json). Code:
[`src/run_phase5_validation_strategy.py`](../src/run_phase5_validation_strategy.py).

## 1. The four protocols, head to head

| Protocol | RMSE mean | RMSE std | Fits required | Wall time |
|---|---|---|---|---|
| RepeatedKFold(5,10) [Phase 3/4 baseline] | 16.693 | 2.235 | 50 | 25.9s |
| Monte Carlo CV (50× random 80/20 splits) | 16.258 | 2.450 | 50 | 25.7s |
| Bootstrap (50× resample, out-of-bag eval) | **18.990** | 2.048 | 50 | 26.3s |
| Nested CV (5 outer × 3-inner-fold grid search) | 16.664 (outer, honest) | 2.564 | **180** | 26.6s\* |

\*Nested CV's wall time is not directly comparable to the others — its 180 fits ran with
`n_jobs=-1` (parallelized grid search), while the other three ran single-threaded. The
**fit count**, not wall time, is the fair cost comparison here, and nested CV needs 3.6× more
fits to produce its estimate.

## 2. Variance of the estimate itself (not just fold-to-fold spread within one run)

Fold-to-fold `std` in the table above describes spread *within* a single run of a protocol.
A more decision-relevant question: if we'd gotten unlucky with the random seed, how different
would the final reported number have been? Measured directly for RepeatedKFold by re-running
the full 5×10 protocol under 20 independent base seeds:

**Mean RMSE across 20 reseeded runs: std = 0.242, range = [16.65, 17.53]** — a ~1.4%
relative wobble on a ~16.7 base. This is a small, practically reassuring number: RepeatedKFold
at n=150 does not appear to be dangerously seed-sensitive for this model. (Scope note: this
reseed-stability check was run only for RepeatedKFold, not the other three protocols, given
the time budget for this phase — a reasonable gap to close later if a specific alternative
protocol becomes the leading candidate for something else.)

## 3. Susceptibility to optimistic bias — a genuinely counter-intuitive result

The textbook story: if you select the best hyperparameter configuration using a validation
split and then report *that* validation score as your generalization estimate, it is
optimistically biased — you specifically chose the configuration that looked best on that
data. Nested CV exists to correct this by re-evaluating the winning configuration on data the
selection process never saw.

**Measured result**: the naive (non-nested) inner-CV-selected score was **19.201**, while the
honest outer (nested) estimate was **16.664** — the naive number is *worse* (higher RMSE),
the opposite direction from the textbook prediction of optimistic bias.

**Why, investigated rather than left unexplained**: the inner CV here trains on a 3-fold split
of the *outer training fold* (120 rows → ~80 rows per inner-fold training set), a
meaningfully smaller training set than RepeatedKFold's own ~120-row folds. Phase 4's learning
curves (`learning_curve_report.md`) showed ExtraTrees' validation RMSE is still falling
steeply between n=80 and n=120 (23.2→16.7 in the learning-curve data). **The data-size
penalty from training on fewer rows inside the nested inner loop outweighs any optimistic
selection-bias effect at this sample size** — a real, measured, non-textbook finding specific
to how small n=150 is, not a general claim that nested CV's bias-correcting purpose is
unnecessary.

**Practical reading**: this result does not mean the *concept* of guarding against
overfitting the validation process is unnecessary — Core Principle 1 stands. It means that at
n=150, the dominant risk in this specific setup is under-provisioning training data inside
nested resampling, not classic hyperparameter-selection optimism. The mitigation is the same
either way: **never report a hyperparameter search's own internal best score as the final
generalization estimate** — always re-validate independently.

## 4. Bootstrap's pessimistic bias

Bootstrap-OOB reports RMSE=18.99, about 2.3 points worse than RepeatedKFold — expected and
well-documented: a bootstrap resample of size n drawn with replacement contains on average
only ~63.2% unique rows, so each bootstrap "training set" has meaningfully less distinct
information than an 80%-of-n K-fold training split, inflating the OOB error pessimistically.
Bootstrap does have the lowest per-run std (2.048) of the four — a real precision advantage —
but its systematic pessimistic bias makes it unsuitable as the primary reporting metric here
without a bias-correction term (e.g. the .632+ estimator), which was not implemented given
the time budget for this phase.

## 5. Decision: protocol selection, justified quantitatively

**Selected protocol for all Phase 5 reporting and tuned-vs-baseline comparisons:
`RepeatedKFold(5, 10)`** — identical to the Phase 3/4 baseline. Justified by:

1. **Directly measured low seed-sensitivity** (§2: ±0.24 RMSE across 20 reseeds).
2. **Close agreement with the "honest" nested-CV estimate** (16.693 vs. 16.664 — a 0.03 RMSE
   gap, far smaller than either protocol's own fold-to-fold std) — evidence this protocol is
   not meaningfully optimistically biased for this use case, without paying nested CV's 3.6×
   fit-count cost on every comparison.
3. **Directly comparable to every number already reported in Phase 3 and Phase 4** — changing
   protocols now would break every existing baseline comparison.
4. Monte Carlo CV performed similarly but offers no measured advantage over RepeatedKFold and
   is known to produce overlapping (non-independent) test sets across splits; Bootstrap's
   pessimistic bias rules it out as a primary metric.

**Selected protocol for the Optuna per-trial objective (hyperparameter search itself)**:
a **lighter RepeatedKFold** — `(5,3)` for the cheaper-to-fit tree models (ExtraTrees,
RandomForest), `(5,2)` for the more expensive models (CatBoost, GaussianProcess) — not the
full `(5,10)` and not full nested CV. This is a deliberate search-efficiency trade-off (Core
Principle 3): full nested CV inside an 80-trial Optuna search would require on the order of
80 × 180 = 14,400 fits per model, computationally prohibitive within this project's runtime
budget. **The safeguard against overfitting the validation process (Core Principle 1) is not
the per-trial protocol — it is that every hyperparameter configuration Optuna selects is
re-evaluated with the full, independent `RepeatedKFold(5,10)` protocol before any
"tuned beats baseline" claim is made** (`hyperparameter_optimization_report.md`). This mirrors
exactly what §3 showed the honest answer requires: never trust a search process's own
internal best score as the final word.
