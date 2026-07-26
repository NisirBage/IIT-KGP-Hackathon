# Phase 5 — Hyperparameter Optimization Report

Framework: [`src/optimization/`](../src/optimization/) (`search_spaces.py`, `objective.py`,
`optimizer.py`, `callbacks.py`, `analysis.py`, `configs.py`). Every study is Optuna
TPE-sampled, SQLite-checkpointed per trial (`artifacts/optuna_studies/<model>.db` — resumable
by construction, which mattered given this environment's history of killing long background
jobs without warning), and MedianPruner-pruned using fold-level intermediate reporting.
Raw results: [`phase5_analysis_results.json`](phase5_analysis_results.json).

## Search protocol (justified in `validation_strategy_report.md`)

Optuna's per-trial objective uses a **lighter** `RepeatedKFold` than the final reporting
protocol — `(5,3)` = 15 folds for ExtraTrees/RandomForest, `(5,2)` = 10 folds for
CatBoost/GaussianProcess (deliberately lighter given their higher per-fit cost). This is a
search-efficiency trade-off, not the number used for any tuned-vs-baseline claim — every
model's Optuna-selected configuration is independently re-evaluated with the full
`RepeatedKFold(5,10)` protocol before being compared to its Phase 4 baseline (see
`optimization_diagnostics_report.md` and `final_model_selection_report.md`).

## Trial counts (scaled down from the phase's "100-200" guidance — see `configs.py`)

| Model | Target trials | Completed | Pruned | Total search time | Mean trial time |
|---|---|---|---|---|---|
| ExtraTrees | 80 | 51 | 29 | 572s (9.5 min) | 11.2s |
| RandomForest | 80 | 59 | 21 | 865s (14.4 min) | 14.7s |
| CatBoost | 60 | 34 | 26 | 493s (8.2 min) | 14.5s |
| GaussianProcess | 25 | 17 | 8 | 79s (1.3 min) | 4.7s |

Trial counts were reduced from the phase's "100-200 trials" guidance after a 5-trial smoke
test measured ~13s/trial for RandomForest — 150 trials would have needed 30+ minutes per
tree model alone. Pruning meaningfully reduced wasted compute (29-45% of trials pruned for
every model) — clearly bad configurations are cut after just 2 folds rather than running the
full objective CV budget.

## Search spaces and rationale

Full rationale (one paragraph per parameter) lives in
[`src/optimization/search_spaces.py`](../src/optimization/search_spaces.py) docstrings —
summarized here. Every space was built from a specific Phase 4 finding, not generic defaults:

- **ExtraTrees / RandomForest**: centered on *regularization* directions (`max_depth` down
  to 3, `min_samples_leaf` up to 15, restricted `max_features`) because Phase 4's learning
  curves showed both models — ExtraTrees especially — reach near-zero training error even at
  the smallest tested training size, a clear high-variance signature.
- **CatBoost**: `l2_leaf_reg` searched up to 20 (vs. default 3.0) for the same
  overfitting-diagnosis reason; `iterations` capped at 1500 and paired with native early
  stopping (a validation slice carved out of each fold's training data, 50-round patience)
  so trial cost doesn't scale linearly with the iteration search range.
- **GaussianProcess**: search spans kernel *family* (RBF, Matern-1.5, Matern-2.5,
  RationalQuadratic) specifically because Phase 4 flagged GP's worst-in-class physical
  plausibility (16% of predictions outside [0,100]) and noisy small-sample training
  behavior — a different kernel's smoothness assumption was a directly motivated hypothesis
  for both problems, not an arbitrary addition.

## Best hyperparameters found

| Model | Best params (Optuna) | Objective value (light CV) |
|---|---|---|
| ExtraTrees | `n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=0.75, bootstrap=False` | 17.044 |
| CatBoost | `depth=7, learning_rate=0.0110, l2_leaf_reg=1.53, iterations=1500, random_strength=0.40, bagging_temperature=1.53` | 17.544 |
| RandomForest | `n_estimators=400, max_depth=12, min_samples_leaf=1, min_samples_split=2, max_features=0.5` | 19.135 |
| GaussianProcess | `kernel=Matern(nu=1.5), length_scale=1.09, noise_level=0.048, n_restarts_optimizer=9` | 20.417 |

**Note on ExtraTrees specifically**: the winning configuration differs from the Phase 4
default in only two of six dimensions (`n_estimators` 200 vs. 300, `max_features` 0.75 vs.
1.0) — every other suggested parameter landed back at the default value. This small a
perturbation, combined with what the final re-validation found (§ below), turns out to be
consequential — see `final_model_selection_report.md`.

**Note on GaussianProcess**: `length_scale` and `noise_level` are only *initial values* for
sklearn's `GaussianProcessRegressor` — by default, `.fit()` further optimizes the kernel's
own hyperparameters internally (gradient-based, plus `n_restarts_optimizer` random restarts)
starting from whatever Optuna suggests. This means Optuna is not directly controlling the
final fitted length-scale/noise the way it directly controls, say, CatBoost's `depth` — it is
searching over kernel *family* and *restart count* more directly than over the continuous
values, which is consistent with `kernel_family` dominating the parameter-importance ranking
(§ `optimization_diagnostics_report.md`) by a wide margin.

## Result headline (full detail in `final_model_selection_report.md`)

| Model | Baseline RMSE | Tuned RMSE (full re-validation) | Change | Statistically significant? |
|---|---|---|---|---|
| **ExtraTrees** | 16.693 | **16.871** | **+0.178 (worse)** | **Yes** (p=0.046 / p=0.025) |
| CatBoost | 17.987 | 17.207 | −0.781 (better) | Yes (p<0.0001) |
| RandomForest | 19.926 | 19.324 | −0.601 (better) | Yes (p=0.002 / p=0.003) |
| GaussianProcess | 20.550 | 19.734 | −0.816 (better) | Yes (p<0.0001) |

**3 of 4 models genuinely improved from tuning. ExtraTrees — the single best baseline model
— got statistically significantly *worse*.** This is the central, sobering finding of Phase 5
and is unpacked fully, including why, in `final_model_selection_report.md`.
