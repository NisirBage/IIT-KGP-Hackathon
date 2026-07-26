# Final Presentation Outline

14 main slides + 4 backup slides. Every main slide is tagged with the narrative beat it
serves (`presentation_story.md`) — nothing here is decorative. 13 figures used across the
main deck, within the 12-15 limit.

## Main deck

**1. Title / framing** (Beat 0)
"Reactor Yield Prediction: An Engineering Investigation, Not a Model Search."
State the central narrative up front: physics → evidence → simplicity, in that order.

**2. The problem & the physics hypothesis** (Beat 1)
Series reaction A→B→C, why yield should be non-monotonic in residence time and
threshold-like in temperature. *Figure*: a simple reaction-network diagram (built directly
in slide software, not a data plot).

**3. Verifying the zero-yield regime statistically** (Beat 2)
25% of training rows are exactly zero — a genuine structural regime, not noise. *Figure*:
`figures/target_distribution.png` (the log-scale panel makes the point cleanly).

**4. The residence-time / thermal interaction** (Beat 2)
The clearest visual evidence of the physics hypothesis. *Figure*:
`figures/interaction_residence_deltaT.png`.

**5. Feature engineering: cast wide, prune hard** (Beat 3)
24 candidates → 5 validated. Headline: `avg_temp` is sigmoidal, not linear — a correction we
made mid-analysis. *Figure*: `figures/phase2/nonlinearity_lowess_top8.png`.

**6. Redundancy discipline** (Beat 3/7)
24 candidates collapse into 6 clusters; more features actively hurt performance in
benchmarking. *Figure*: `figures/phase2/feature_dendrogram.png`.

**7. Leakage-free pipelines, proven not assumed** (Beat 4)
Every transform lives inside a refit-per-fold Pipeline. *Figure*: a simple pipeline-flow
diagram (built in slide software from the architecture in `submission_pipeline_report.md`) —
optionally paired with `figures/train_test_comparison.png` if train/test-shift context is
useful for this audience.

**8. Model comparison: statistics over leaderboard-reading** (Beat 5)
13 models, identical protocol, Friedman + Nemenyi. *Figure*: `figures/phase8/leaderboard.png`.
Callout: Nemenyi overturns a naive "ExtraTrees beats CatBoost" read (p=0.953 corrected vs.
p=1.5e-8 uncorrected) — say this explicitly, it's a strong credibility signal.

**9. Cross-model agreement on physical drivers** (Beat 2/5)
`avg_temp`/`jacket_temperature_K` dominate every model family; `concentration_mol_L` is
consistently irrelevant — ML confirms the chemistry. *Figure*:
`figures/phase4/feature_importance_comparison.png`.

**10. Tuning: caught our own mistake** (Beat 6)
Optuna improved 3 of 4 shortlisted models — and made ExtraTrees *worse*, caught by mandatory
re-validation. This is the strongest "our process works" moment — spend real time here.
*Figure*: `figures/phase5/convergence_CatBoost.png` (the tuning success story) — narrate the
ExtraTrees regression verbally/via table, since its story is about a number moving the wrong
way, not a convergence curve.

**11. Ensembling: skepticism first, evidence second** (Beat 7)
Correlation alone said "no potential" (≥0.92 among all models). Disagreement-rate analysis
said otherwise. A blend won — but a red-flag coefficient pattern meant we didn't trust it
until 10 independent robustness checks and a full nested validation confirmed it. *Figure*:
`figures/phase5/tuned_model_diversity.png`.

**12. The result** (Beat 9)
`figures/phase8/ensemble_improvement.png` — both panels: the headline RMSE comparison and the
region-specific breakdown showing the gain is broad-based, not a lucky region.

**13. Reproducibility & lockdown** (Beat 8)
Git-tagged, hash-verified, deterministic (3/3 identical runs), adversarially tested (found
and fixed a real bug). One-command submission generation. No new figure needed — reuse the
pipeline diagram from slide 7 if a visual anchor is wanted.

**14. What could still go wrong** (Part F — see dedicated section below)

## Part F — Failure modes (own section within the main deck, slide 14, expandable to 2 slides if time allows)

Four honest, named risks, framed as "here's what we'd want to know before trusting this in
production," not apologetically:

- **Distribution shift**: PSI flagged a moderate train/test shift signal on `length_m`
  specifically in Phase 1 — never fully resolved, only defended against via robust
  (repeated) validation rather than eliminated.
- **Extrapolation**: tree-based models cannot extrapolate beyond training ranges by
  construction — any operating condition outside the training envelope is a blind spot, not
  a graceful degradation.
- **Unseen operating regimes**: the ensemble's coefficients (including a negative weight on
  RandomForest) are validated in-distribution only — no test exists against genuinely
  different future data (`confidence_audit.md` row 18).
- **Uncertainty**: the shipped model outputs a point estimate only — no calibrated
  uncertainty band, despite the underlying per-sample variance data already existing in our
  diagnostics (`future_work.md` medium-term).
- **Maintenance**: this model is specific to this reactor's chemistry — a design or catalyst
  change requires retraining from scratch, not fine-tuning.

*Figure (optional, if the panel is technical)*: `figures/phase4/residual_diagnostics_top6.png`
— shows every model has heteroscedastic, right-skewed residuals; a shared, honestly-reported
limitation, not unique to the final model.

## Backup slides (not presented unless asked)

**B1. Why are many predictions exactly zero?** *(explicitly requested — keep this ready, not
buried)*

- **The observation**: 21 of 50 test predictions (42%) are clipped to exactly 0.0, compared
  to a 24.7% zero-yield rate in the 150-row training set.
- **Two possible explanations**, neither provable without ground-truth test labels:
  (a) the test set's operating conditions sample the high-temperature/long-residence
  "collapse" region more heavily than training did; (b) the model is systematically
  defaulting to the "safe," physically-valid floor when genuinely uncertain, since clipping
  makes 0 a very reachable value for any negative raw prediction.
- **The evidence that favors (a) over (b)**: test rows predicted as zero have a mean
  `avg_temp` of 467.6K — almost exactly matching the training set's zero-yield-group mean of
  469.1K (a 1.5K gap, against a feature with ~36K standard deviation). This is strong,
  specific evidence the model is recognizing the *same* physically-defined collapse regime
  identified throughout Phases 1-2, not behaving arbitrarily. (One caveat, stated honestly:
  `delta_T` for these same rows does *not* match the training zero-yield group's pattern as
  closely — plausibly because these particular test rows reach high `avg_temp` via a
  different inlet/jacket combination than most training zero-yield rows did, while still
  landing in the same "hot" regime.)
- **Why clipping was applied**: directly shown to *improve*, not just constrain, accuracy —
  clipped LOO RMSE (14.762) beats unclipped (14.974), because a negative prediction for a
  true near-zero row becomes exactly correct once clipped.
- **What would distinguish the two explanations**: the true test-set labels (unavailable to
  us) would settle this immediately. Short of that, the single most informative next
  experiment would be checking whether the test set's `avg_temp` distribution, specifically
  among rows the model flags as zero, differs from a *held-out subset of the training set's*
  zero-yield rows in any measurable way — if they look statistically indistinguishable (which
  our one check suggests), that's the strongest evidence available for explanation (a)
  without new data.

**B2. Confidence audit summary** — pull the 5-6 highest-stakes rows from
`confidence_audit.md` (rows 1, 9/10, 16, 18, 22) as a single reference slide if the panel
wants a rapid-fire "what are you sure of" answer.

**B3. Validation protocol comparison** — the RepeatedKFold vs. Nested vs. Monte Carlo vs.
Bootstrap comparison table from `validation_strategy_report.md`, for a statistics-heavy
panel that wants to interrogate the protocol choice specifically.

**B4. Full model comparison table** — the complete 13-model leaderboard with MAE/MedAE/R²/
runtime columns (`baseline_model_report.md`), for anyone who wants the raw numbers behind
slide 8's chart.
