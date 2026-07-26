# Presentation Story

## The central narrative

**Not**: "We tried many models and picked the best one."

**Instead**: "We treated this as an engineering investigation — understand the physics
first, let evidence (not intuition, and not any single number) drive every decision, and
ship the simplest thing that survives every test we could throw at it."

This framing matters because it's *true*, not just good marketing — the project's own
history repeatedly shows evidence overturning the "obvious" choice: the best-looking model
degraded under tuning (Phase 5); the "correlated models can't ensemble" intuition was wrong
once measured properly (Phase 6); a feature that looked redundant by correlation turned out
to carry the strongest incremental signal in the project (Phase 2). The narrative isn't
aspirational — it's a description of what actually happened, which is why it will hold up
under questioning.

## The nine-beat structure

Every slide should trace back to one of these beats. If a slide doesn't obviously serve one
of them, cut it.

1. **Understand reactor physics** — series reaction A→B→C, why residence time and
   temperature should behave non-monotonically/non-linearly, *before* looking at data.
2. **Verify hypotheses statistically** — don't assume the physics story is right; test it.
   The zero-yield gap, the non-monotonic MI signature, the sigmoidal temperature threshold —
   each is a physics hypothesis confirmed (or refined) by a specific statistical test.
3. **Engineer physically meaningful features** — 24 candidates generated from theory, 5
   survived rigorous validation (correlation battery, nested F-tests, redundancy analysis).
   Cast a wide net, prune hard.
4. **Build leakage-free pipelines** — every transform lives inside a `sklearn.Pipeline`,
   refit per fold. Proven, not assumed: leaky preprocessing is measurably optimistic in
   29/30 trials.
5. **Compare models scientifically** — 13 models, identical protocol, Friedman/Nemenyi
   correction — not just a leaderboard sort.
6. **Tune only promising candidates** — 4 shortlisted families, not all 13. And when tuning
   made the best model *worse*, the project caught it and reverted, rather than trusting the
   tuning process blindly.
7. **Reject unnecessary complexity** — weighted averaging (rejected, no gain), the
   4th ensemble member (rejected, p=0.615 to drop), regularized stacking vs. plain blending
   (statistically identical, kept the simpler one).
8. **Validate reproducibility** — byte-identical output across repeated runs, adversarial
   testing that found and fixed a real bug, full artifact hashing.
9. **Ship the simplest model that survives every test** — not the most complex model built,
   not the single best-looking baseline, but the 3-model blend that beat every statistical
   and physical-plausibility check thrown at it.

## What makes this story credible, not just well-organized

Three moments in the project are the strongest proof points that this was genuine
investigation, not retrofitted narrative — lead with these when time is short:

- **The tuning reversal** (beat 6): ExtraTrees, the best baseline model, got *statistically
  significantly worse* after hyperparameter tuning. The project's own validation discipline
  caught this and the final recommendation follows the evidence, not the "tuning phase's"
  output. This is the single best demonstration that the process is trustworthy — it worked
  even when it meant contradicting the project's own prior best result.
- **The ensemble stress-test** (beat 7→9): a surprising ~12% RMSE improvement from blending
  showed a coefficient pattern (negative weight on RandomForest) that closely resembled a
  known failure mode. It was not reported until stress-tested 10 independent ways and
  validated with a full nested cross-validation.
- **The validator bug** (beat 8): adversarial testing of the frozen pipeline found a real
  crash bug in the submission validator before it ever reached judges — direct proof the
  "reproducibility validation" step wasn't a checkbox exercise.

## Slide-by-slide narrative discipline

For each slide in `presentation_outline.md`, ask: *which of the 9 beats does this serve, and
would removing it break the story's logical chain?* If a result doesn't change what beat 9
looks like, it's supporting evidence for an earlier beat, not a headline in its own right —
demote it to backup material (see the backup-slide section of the outline).
