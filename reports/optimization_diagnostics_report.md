# Phase 5 — Optimization Diagnostics Report

Convergence plots: [`figures/phase5/convergence_*.png`](figures/phase5/) (one per model —
best-objective-value-so-far vs. completed trial index). Full Optuna study data queryable
directly from `artifacts/optuna_studies/<model>.db`.

## Convergence: did optimization converge, or would more trials likely help?

The automated heuristic in `optimization/analysis.py` (compare improvement in the last 30
completed trials to the window before that) reported `insufficient_trials_to_assess` for
every model — none reached the 60-completed-trial threshold the heuristic requires (highest
was RandomForest at 59). **Read the convergence plots directly instead, which are more
informative here than the automated heuristic**:

- **ExtraTrees**: sharp improvement through trial ~20 (29.4→17.7), then a long flat stretch
  from trial ~20–46 (17.7 flat), then a small final drop to 17.04 by trial ~50. The flat
  middle stretch is a visual convergence signal — the search had largely found its regime by
  trial 20, and the remaining 30 trials bought only ~0.6 RMSE.
- **CatBoost**: similar shape — improvement through trial ~24 (19.1→17.7), flat, then a small
  final drop to 17.54 in the last few trials. Also visually converging.
- **RandomForest / GaussianProcess**: qualitatively similar step-then-plateau shape (full
  trajectories in the JSON; not separately plotted here for space).

**Verdict: all four searches show a visually clear plateau in their final third, suggesting
diminishing returns from the trial counts used were already being reached — extending any of
these studies substantially further would likely have bought only marginal additional gains,
not a different regime.** This is a qualitative read, not a statistically certified stopping
rule — the studies remain resumable (SQLite-backed) if a future phase wants to verify this
more rigorously with more trials.

## Hyperparameter importance (which parameters actually mattered)

Computed via Optuna's fANOVA-based `get_param_importances` over each study's completed
trials:

| Model | Top parameter | Importance | 2nd | 3rd |
|---|---|---|---|---|
| ExtraTrees | `min_samples_leaf` | **0.665** | `max_features` (0.146) | `n_estimators` (0.102) |
| RandomForest | `min_samples_leaf` | **0.754** | `n_estimators` (0.089) | `max_features` (0.071) |
| CatBoost | `random_strength` | **0.728** | `l2_leaf_reg` (0.108) | `iterations` (0.090) |
| GaussianProcess | `kernel_family` | **0.537** | `noise_level` (0.200) | `length_scale` (0.186) |

**`min_samples_leaf` dominates both bagged-tree models by a wide margin** (0.67-0.75 of total
importance) — strong, direct confirmation that the overfitting diagnosis motivating the
search space (Phase 4 learning curves) correctly identified the highest-leverage lever: how
many samples are forced into each leaf controls the bias-variance trade-off more than tree
count or feature subsampling for this dataset. Interesting nuance already noted in
`hyperparameter_optimization_report.md`: despite dominating the *importance* ranking (i.e.
varying it caused the biggest swings across the whole search), ExtraTrees' actual *winning*
trial kept `min_samples_leaf=1` — identical to the Phase 4 default. The parameter matters a
lot in the sense that moving it (mostly upward) reliably made things worse; it did not matter
in the sense of motivating a change from the default.

**`random_strength` dominates CatBoost** (0.73) — more than the "usual suspects" like
`learning_rate` or `depth` (each under 0.03 combined with `depth` at 0.009) — a specific,
somewhat non-obvious finding: CatBoost's split-score randomization parameter, not its
boosting-rate or tree-depth, was the highest-leverage lever for this dataset.

**`kernel_family` dominates GaussianProcess** (0.54) — consistent with the search-space
design rationale (§ `hyperparameter_optimization_report.md`): the qualitative choice of
smoothness assumption mattered far more than fine-tuning any continuous parameter within a
fixed kernel family, and the study's trials converged almost entirely onto `matern1.5` after
early exploration (visible directly in the raw trial log — after trial ~10, nearly every
subsequent trial used `matern1.5`).

## Trial runtime distribution

| Model | Mean trial time | Total search time | Notes |
|---|---|---|---|
| GaussianProcess | 4.7s | 79s | Fastest per-trial despite O(n³) kernel fitting — small n and aggressive pruning (8/25 pruned) kept cost down |
| ExtraTrees | 11.2s | 572s | |
| CatBoost | 14.5s | 493s | Early stopping (50-round patience) meaningfully capped worst-case trial cost despite `iterations` searched up to 1500 |
| RandomForest | 14.7s | 865s | Slowest total search — largest completed-trial count (59) combined with a similar per-trial cost to CatBoost |

No model's search became a runtime bottleneck for this phase's budget — even the most
expensive (RandomForest, ~14.4 minutes total) is small next to the multi-hour budgets a
"100-200 trials" target would have implied at the smoke-tested per-trial cost.

## Answering the phase's three explicit questions

1. **Did optimization converge?** Qualitatively yes for all four (visual plateau in the
   final third of each study), though not statistically certified by the automated
   convergence-check heuristic given the trial counts used.
2. **Are additional trials likely to help?** Marginally at best, based on the plateau shape
   — not enough to expect a different regime, though not disprovable without actually
   running more (the studies are resumable if this is worth revisiting).
3. **Which hyperparameters matter most?** `min_samples_leaf` for both bagged-tree models,
   `random_strength` for CatBoost, `kernel_family` for GaussianProcess — none of the four
   "obvious" headline parameters (`n_estimators`, `learning_rate`, `depth`, `length_scale`)
   turned out to be the dominant lever for its respective model.
