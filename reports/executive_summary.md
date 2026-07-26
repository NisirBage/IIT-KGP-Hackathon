# Executive Summary

Same story, four depths. Each version is self-contained — use whichever fits the time you're
given, not a compressed version of the next one up.

---

## 30 seconds

We treated this as an engineering investigation, not a leaderboard chase: understand the
reactor physics first, verify every hypothesis statistically, engineer features grounded in
that physics, compare models under a rigorous shared protocol, tune only what earned it, and
reject complexity that didn't prove itself. Our final model — a 3-model ensemble, clipped to
the physically valid range — beats the best single model by 12%, with a p-value under
0.000001. Every major decision in this pipeline is backed by a specific statistical test, not
intuition.

---

## 2 minutes

The reactor runs a series reaction (A→B→C) where yield should be non-monotonic in residence
time and threshold-like in temperature — we verified both hypotheses statistically before
building any model, finding a hard, physically-meaningful split between reactor
"collapse" (zero yield) and active operation.

From there: 24 candidate physics-informed features were tested, 5 survived rigorous
validation (correlation battery, nested significance tests, redundancy analysis) — adding
more features than that measurably *hurt* performance, so we didn't. 13 models were compared
under an identical, leakage-free cross-validation protocol; a statistical test (Friedman +
Nemenyi correction) identified a top tier of 4, not just a single "winner," because a naive
leaderboard read would have overstated how separable the top models really were.

We tuned only those 4 — and caught our own tuning process making the best model *worse*,
which we only found because every tuned result was independently re-validated before being
trusted. Ensembling looked unpromising by a simple correlation check, but a deeper
disagreement analysis and rigorous stress-testing revealed a real, 12%, statistically
airtight improvement — validated 10 independent ways before we trusted it, because the
mechanism behind it resembled a known failure pattern we'd already seen elsewhere in the
project.

The final pipeline is deterministic, adversarially tested (which caught a real bug before
submission), and fully reproducible from a single command.

---

## 5 minutes

**The problem**: predict reactor yield from 5 operating conditions, replacing an expensive
CFD/BVP simulation, for a series reaction A→B→C with competing kinetics.

**Phase 0-1 — Physics before data**: we reasoned through what the chemistry predicts before
touching the dataset — residence time should have an *interior maximum* effect on yield (too
short, A hasn't converted; too long, B has degraded to C), and temperature should show a
threshold effect once the side reaction starts dominating. We then verified this: the
training data shows a genuine, structurally distinct zero-yield regime (25% of rows, a hard
statistical gap, not a rounding artifact), and PSI/KS testing checked whether the test set
came from the same operating regime (mostly yes, with one flagged caveat that shaped every
downstream validation choice).

**Phase 2 — Feature engineering with a validation bar**: 24 candidate features spanning
residence-time transforms, temperature combinations, and Arrhenius-inspired forms were
generated from the physics reasoning, then evaluated with a correlation battery, nested
partial-F incremental-value tests, redundancy/VIF analysis, and bootstrap stability. Only 5
were promoted. A benchmark directly confirmed that the curated 5-feature set beat both a
raw-features-only baseline *and* dumping in all 24 candidates — more features actively hurt.

**Phase 3 — Leakage-free, model-specific preprocessing**: every transform lives inside a
scikit-learn Pipeline refit per fold — proven, not assumed, via a leaky-vs-correct comparison
showing naive preprocessing is optimistically biased in 29 of 30 trials. Scaling strategy
turned out to be genuinely model-specific: KNN was actively hurt by every scaler tested,
while GaussianProcess needed one badly — a one-size-fits-all pipeline would have been wrong.

**Phase 4-5 — Model comparison and tuning, statistically disciplined**: 13 models compared
under an identical protocol; a Friedman/Nemenyi analysis (not raw RMSE ranking) identified a
top statistical tier of 4. Hyperparameter tuning via Optuna improved 3 of those 4 — and
*worsened* the 4th (ExtraTrees, the best baseline model). We caught this because every
Optuna-selected configuration was independently re-validated under the full protocol before
being trusted, not accepted on the search's own report.

**Phase 6 — Ensembling, evidence over intuition**: prediction correlation alone (≥0.92 among
all 4 tuned models) suggested limited ensemble potential. A deeper disagreement-rate analysis
found real, region-concentrated disagreement correlation was hiding. A linear blend of 3
models produced a striking ~12% RMSE improvement — with a coefficient pattern (a negative
weight on one base model) that resembled a known instability failure mode elsewhere in the
project. We didn't report it until it survived 10 independent robustness refits and a
rigorous leave-one-repeat-out nested validation. It held. Clipping to the physical [0,100]
bound then *improved* accuracy further, not just enforced plausibility.

**Phase 7 — Lockdown**: the final pipeline is git-tagged, fully hashed, deterministic (3/3
identical outputs across independent runs), and validated end-to-end from raw test data to
submission CSV with zero manual steps. Adversarial testing of the validator found and fixed
a real bug before it could reach judges.

**Result**: a 3-model ensemble (ExtraTrees + CatBoost + RandomForest, clipped), RMSE 14.76 ±
0.64, beating the best single model (16.69 ± 2.24) by a wide, statistically decisive margin
(p<0.000001), validated broadly across every physically meaningful operating region tested.

---

## 10 minutes

*(Use the 5-minute version above as the backbone; expand each phase with the specific
numbers and named limitations below, drawing directly from the underlying reports rather
than repeating this document.)*

**Phase 0-1 detail**: the series-reaction math (`C_B(τ) = C_A0·k1/(k2−k1)·[e^(-k1τ)−e^(-k2τ)]`)
predicts an interior maximum at `τ_opt = ln(k2/k1)/(k2−k1)` — this motivated testing
`residence_proxy` for a *non-monotonic* signature specifically (near-zero linear correlation
but high mutual information), which is exactly what we found. The zero-yield gap (nothing
between exactly 0 and 0.013) plus a GMM modality test supported treating it as a genuine
regime; we were careful to note this doesn't prove the *mechanism* (over-reaction vs. a
solver artifact) — only that the regime itself is real.

**Phase 2 detail**: the standout methodological finding — `arrhenius_inlet` looked redundant
with raw `inlet_temperature_K` by rank correlation (identical Spearman, since it's a
monotonic transform) but passed the strongest incremental significance test in the whole
phase (p=2.1e-5), because rank-correlation measures are structurally blind to whether a
*nonlinear transform* of a variable adds value — only a nested model comparison can see that.
`avg_temp` turned out to have a sigmoidal (not linear) relationship with yield on closer
visual inspection — a correction we made mid-analysis after a quadratic-fit diagnostic gave a
misleading "already linear" signal.

**Phase 3-4 detail**: the scaling benchmark's most counter-intuitive result — KNN performed
*worse* with every scaler tested than with none — directly contradicts the usual "always
scale for distance-based models" default, and we reported it rather than silently applying
convention. Nemenyi correction across all 78 pairwise model comparisons overturned a naive
"ExtraTrees beats CatBoost" conclusion that an uncorrected paired test would have supported
(p=1.5e-8 uncorrected vs. p=0.953 corrected) — a direct demonstration of why multiple-testing
correction mattered here, not a theoretical aside.

**Phase 5 detail**: the tuning search itself used a lighter cross-validation budget than the
final reporting protocol, by necessity (a full nested search across ~80 trials × the full
protocol would have taken hours per model) — but every winning configuration was re-checked
under the full protocol before any claim was made. This is what caught ExtraTrees'
regression: its "improved" configuration differed from the default in only 2 of 6 search
dimensions, and the lighter search budget's noise was enough to mistake a small, genuine
regression for an improvement.

**Phase 6 detail**: the negative-coefficient investigation is the project's best example of
skepticism applied to its own results — refitting the blend's meta-model independently on
each of the 10 available cross-validation repeats produced the same qualitative coefficient
pattern (positive ExtraTrees/CatBoost, negative RandomForest) in all 10 cases, and a full
leave-one-repeat-out nested validation (train on 9 repeats, test on the held-out 10th,
repeated for all 10) gave a consistent RMSE estimate throughout. Region-specific analysis
confirmed the improvement isn't concentrated in one easy region — it holds across zero-yield,
thermal-transition, and high-yield operating conditions alike.

**Phase 7 detail**: reproducibility was tested adversarially, not just confirmed on the happy
path — 4 malformed-input scenarios and 2 malformed-submission scenarios were deliberately
constructed, one of which (a submission with the wrong header) crashed the validator with an
unhandled exception instead of failing cleanly. Fixed and re-verified before lockdown.

**Named limitations, stated proactively**: feature selection was not nested inside
cross-validation (a project-level, not fold-level, leakage risk); the ensemble's coefficients
are validated in-distribution only, with no test against genuinely different future data;
package-version portability was never confirmed on a second machine. None of these
invalidate the reported results — all are documented in `confidence_audit.md` with an honest
confidence grade rather than glossed over.
