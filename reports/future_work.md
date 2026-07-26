# Future Work

Three tiers, each explicitly graded on whether evidence from this project actually supports
the claim that it would help — per the phase instruction, nothing here is presented as a
guaranteed leaderboard improvement unless this project's own data says so.

## Short-term (days, no new data required)

| Idea | Evidence this would help | Confidence |
|---|---|---|
| Re-run the ensemble evaluation with `abs_delta_T` / `arrhenius_inlet` (Phase 2's "validated-optional" features) included in the base models | Untested combination — these features passed individual incremental-value tests (Phase 2 §3) but were never fed into the specific 3 base models used in the final ensemble | **Speculative** — plausible, no direct evidence either way |
| Add an out-of-distribution / extrapolation check to the inference pipeline (flag predictions where input features fall outside the training range) | Directly motivated by the project's own named limitation (`technical_defense.md` Q14, Q61) — not yet built | **Strongly motivated**, not yet evidenced (no implementation exists to measure) |
| Extend the Optuna studies for ExtraTrees/RandomForest/CatBoost beyond their current trial counts (all SQLite-resumable) | The convergence-check heuristic explicitly reported "insufficient trials to assess" for every model; visual plateau suggests diminishing returns, not proven absence of further gain | **Plausible** small gain, likely diminishing given the visual plateau (`optimization_diagnostics_report.md`) |
| Sweep the `ARRHENIUS_C` constant in `arrhenius_inlet`/`arrhenius_avg` instead of using the arbitrary fixed value of 1000 | The feature already shows the strongest incremental-value result in Phase 2 (p=2.1e-5) at an unfit constant — a properly fit constant could plausibly strengthen it further, or reveal the arbitrary choice was already near-optimal | **Plausible**, untested |

## Medium-term (weeks, would benefit from a small amount of new data or moderate new engineering)

| Idea | Evidence this would help | Confidence |
|---|---|---|
| Collect more training data, even a modest amount (e.g. +50-100 rows) | **Directly evidenced**: none of the 4 top models' learning curves had plateaued at n=120 (`learning_curve_report.md`) — validation RMSE was still falling at the largest training size tested | **Strongly supported** — the single most evidence-backed recommendation in this document |
| Nest feature selection inside the outer CV loop (address the Phase 2 meta-level leakage concern named in `technical_defense.md` Part A) | Directly motivated by a named, unresolved project limitation — no experiment yet quantifies how large this effect actually is | **Strongly motivated**, unmeasured magnitude |
| Add prediction-uncertainty quantification (e.g. quantile regression, conformal prediction wrapper, or surfacing the existing fold-to-fold prediction std as a confidence band) | The underlying signal already exists and was computed for diagnostics (`residual_analysis_report.md` §3) but never exposed to the end user/deployment layer | **Plausible**, straightforward given existing infrastructure |
| Evaluate a beta-regression or logit-transformed-target model family, which would respect the [0,100] bound by construction rather than via post-hoc clipping | Clipping was shown to *help*, not just constrain (`ensemble_evaluation_report.md` §6) — a model with the bound built in might capture that benefit even more directly, but this is untested against the current approach | **Speculative** |
| Independently verify the frozen pipeline in a second, freshly-created environment (not just the one venv used throughout) | Directly motivated by a named Medium risk in `competition_readiness_report.md` §2 | **Strongly motivated**, zero risk to attempt |

## Industrial-scale roadmap (months+, would require real plant data and deployment infrastructure)

| Idea | Evidence this would help | Confidence |
|---|---|---|
| Retrain against real plant operating data (not simulator output) | This project's entire feature/model selection was validated against a CFD/BVP *simulator's* output — real plant data would have measurement noise, sensor drift, and potentially different operating envelopes never seen here | **Necessary step, not optional** — no current evidence this model transfers to real plant data at all |
| Build the drift-monitoring and OOD-detection system sketched in `technical_defense.md` Q56/Q61 | Directly motivated by the project's own risk analysis; no such system currently exists even as a prototype | **Strongly motivated**, unbuilt |
| Explore a hybrid mechanistic + ML model (use the ML surrogate for fast optimization, periodically cross-check against the true CFD/BVP simulation) | Consistent with Phase 0's explicit reasoning for why a pure mechanistic refit was rejected (unknown kinetic parameters) while acknowledging a pure black-box model forfeits physical guarantees a hybrid could partially restore | **Plausible**, architecturally reasonable, unbuilt and unevaluated |
| Real-time closed-loop integration with plant control systems, including the safety guardrails named in `technical_defense.md` Q61 | Out of scope for this project entirely — no infrastructure, no real-time data feed, no control-system integration exists | **Necessary for the stated end goal**, entirely future work |

## What this project does *not* claim

None of the above is claimed to improve this project's specific CV-measured RMSE without
first being tested — several (more data, nested feature selection, environment verification)
are strongly motivated by named limitations rather than by any experiment showing they'd move
the number. This distinction is deliberate: the project's own standard throughout has been
"don't claim a finding until it's tested," and future work is held to the same bar.
