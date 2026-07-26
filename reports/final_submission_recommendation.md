# Phase 6 — Final Submission Recommendation

## Decision: Ship a blended ensemble (Option 2/3 hybrid — a weighted linear blend, architecturally identical to the suggested stack)

**Ship the 3-model linear blend: ExtraTrees + CatBoost + RandomForest → fixed linear
combination, clipped to [0,100].** Full derivation and stress-testing in
[`ensemble_evaluation_report.md`](ensemble_evaluation_report.md). This is not "Ship
ExtraTrees" (Option 1) — the exit criterion for that option (no statistically significant
ensemble improvement) was not met. It is functionally the same architecture as the phase's
suggested stacking recipe (§5 of the phase brief) — a linear combination of ExtraTrees/
CatBoost/RandomForest OOF predictions — evaluated here as a blend (`LinearRegression`) since
that is what the evidence supported; a Ridge-regularized stack was also tested and performed
statistically identically (14.967 vs. 14.963 RMSE, §3 of the evaluation report).

## Justification against the five decision criteria

| Criterion | Verdict |
|---|---|
| **Predictive accuracy** | RMSE 14.762 ± 0.642 vs. ExtraTrees' 16.693 ± 2.235 — a 1.93-point (~12%) reduction, the largest single improvement found anywhere in this project since Phase 2's feature engineering |
| **Statistical confidence** | Paired t-test p<0.000001, Wilcoxon p=0.0020, 95% bootstrap CI=[−2.03,−1.85] — clearly excludes zero by a wide margin, not a borderline call |
| **Robustness** | The critical concern (unconstrained linear regression on 4 highly-correlated (0.92-0.99) predictors, producing a *negative* RandomForest coefficient) was stress-tested 10 independent ways (train on each repeat, evaluate on the other 9) and confirmed via a fully rigorous leave-one-repeat-out nested check — the pattern and the RMSE gain are stable across every test, not an artifact of one lucky split. The blend's fold-to-fold std (0.64) is also dramatically *lower* than ExtraTrees' own (2.24) — more consistent, not just more accurate on average |
| **Engineering simplicity** | 3 base pipelines (already-built, already-serialized from Phases 4/5) + one fixed linear combination + a clip — no new model families, no new training infrastructure. GaussianProcess was tested and dropped (p=0.615, zero measurable cost) to keep this as simple as it can be without losing accuracy |
| **Reproducibility** | Blend coefficients are fixed constants derived out-of-fold (never fit on data used to score them), serialized alongside the base pipelines in `models/ensemble.py`'s `LinearBlendEnsemble` — deterministic given the same base pipeline artifacts |

## What tipped this from "reject" to "ship"

The initial blend result (a 1.7-2 point RMSE drop from combining 0.92-0.99-correlated
models) was exactly surprising enough to warrant the same skepticism this project has applied
to every other counter-intuitive number (Phase 1's zero-yield gap, Phase 2's SHAP
disagreement, Phase 3's leakage bias direction, Phase 5's ExtraTrees tuning regression). The
mechanism — a negative RandomForest coefficient — closely resembled a known failure pattern
(Ridge's `residence_sq`/`residence_proxy` collinearity instability, Phase 4). **The
difference between "another collinearity artifact" and "a real, exploitable pattern" was
established empirically, not assumed either way**: 10 independent robustness refits, a
proper leave-one-repeat-out nested validation, and a region-specific breakdown showing
broad-based (not concentrated/lucky) gains all pointed the same direction. That is the bar
this project has held every finding to, and this one cleared it.

## Physical plausibility resolution

The unclipped blend would have shipped with 24.7% physically impossible predictions —worse
than any individual model tested in this entire project. **Post-hoc clipping to [0,100] is
not a workaround being quietly applied to hide a flaw — it is standard, defensible practice
for a model with a known, physically-bounded output range, and it improves (not just fixes)
the RMSE** (14.974 → 14.762), because a negative prediction for a true near-zero row becomes
exactly right once clipped rather than merely closer.

## Deployment artifact

[`artifacts/tuned_pipelines/FINAL_ENSEMBLE_blend_v1.joblib`](../artifacts/tuned_pipelines/)
— a fitted `LinearBlendEnsemble` (3 base pipelines + fixed coefficients + clip), verified via
a serialization round-trip (predictions identical after reload). Coefficients:
`ExtraTrees=1.398, CatBoost=0.928, RandomForest=−1.211, intercept=−5.294`
([`phase6_final_3model_blend.json`](phase6_final_3model_blend.json)).

## What this means for Phases 7-8

Per the user's own proposed next steps: **Phase 7 (submission pipeline)** should generate
predictions using `FINAL_ENSEMBLE_blend_v1.joblib`, not the standalone ExtraTrees model.
**Phase 8 (presentation prep)** gains a genuinely interesting technical story: prediction
correlation alone (Phase 5) said ensembling wouldn't help, but disagreement-rate analysis and
rigorous stress-testing found a real, substantial, statistically bulletproof improvement
anyway — a strong, concrete example of "engineering intuition + statistical rigor beating a
single summary statistic" for the judges' evaluation criteria on process understanding. The
negative-RandomForest-coefficient finding, and why it was trusted only after 10-way
robustness testing, is worth including directly in the technical narrative as evidence this
team validated rather than assumed.

## One honest caveat for the presentation

The blend's coefficients (particularly RandomForest's negative weight) are fit on this
specific dataset's specific model triple and are **not guaranteed to remain optimal** if the
underlying data distribution shifts meaningfully (e.g. genuinely new operating regimes in a
production deployment) — a risk inherent to any learned combination weight, but worth
naming proactively rather than waiting for a judge to ask. A convex (non-negative,
sum-to-one) blend would be more conservative here at a real cost in accuracy (§2 of
`ensemble_evaluation_report.md` — the best convex/weighted-average option was 17.40, far
behind 14.76); this trade-off — maximal validated accuracy vs. extrapolation caution — was
made deliberately in favor of accuracy given how thoroughly the current gain was validated,
and should be stated as a deliberate choice if asked, not discovered as a gap.
