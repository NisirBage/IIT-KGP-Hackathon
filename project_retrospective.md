# Project Retrospective

**Audience**: our future selves, and whoever starts the next industrial ML competition.
Not a judge-facing document. Written the way a Staff ML Engineer writes a postmortem after
a project ships — evidence-grounded, unflattering where warranted, specific rather than
generic.

---

# Part 1 — Executive Retrospective (chronological, by reasoning not by date)

## Phase 0 — Problem Understanding
**Objective**: understand the reactor's chemistry before writing a line of data-analysis
code. **Major decision**: derive testable hypotheses from the series-reaction kinetics
(`A→B→C`) — a non-monotonic residence-time effect, a temperature-driven side-reaction
trade-off — *before* looking at the data. **Evidence used**: the textbook closed-form
solution for consecutive first-order reactions (`τ_opt = ln(k2/k1)/(k2-k1)`), applied purely
as reasoning, not fit to anything yet. **Mistakes discovered**: none — this phase produced
hypotheses, not conclusions, so there was nothing yet to be wrong about. **Outcome**: a
written hypothesis set that every later phase would be graded against, not a vague sense of
"the chemistry probably matters."

## Phase 1 — Dataset Audit
**Objective**: test Phase 0's hypotheses against real data, and establish ground truth about
the dataset's integrity. **Major decision**: treat the 25% exact-zero-yield mass as a
research question, not an assumption — was it a rounding artifact or a genuine regime?
**Evidence used**: a hard gap between exact 0 and the smallest non-zero value (0.013), a
GMM component-count comparison, later cross-checked by classifier separability in Phase 2.
**Mistakes discovered**: none major, but this phase set a durable tone — every subsequent
"interesting number" got the same treatment (verify before you narrate). **Outcome**: a
dataset passport and a PSI-flagged train/test shift concern on `length_m` that shaped the
validation-protocol choice three phases later.

## Phase 2 — Feature Engineering
**Objective**: turn Phase 0's physical reasoning into engineered features, validated
statistically rather than accepted on intuition. **Major decision**: build 24 candidates,
promote only 5. **Evidence used**: a 5-method correlation battery, nested partial-F
incremental-value tests, VIF/hierarchical-clustering redundancy analysis, bootstrap
stability. **Mistake discovered and corrected mid-phase**: a quadratic-R² shape diagnostic
initially reported `avg_temp`'s relationship with yield as "already linear" — visually
inspecting the actual LOWESS curve showed a clear sigmoidal (threshold) shape instead; the
diagnostic was simply the wrong tool for that shape, and the finding was corrected before it
propagated into later phases. **A near-miss avoided**: `arrhenius_inlet` looked redundant by
Spearman correlation (identical to raw `inlet_temperature_K`, since rank correlation is
invariant to monotonic transforms) — a nested F-test caught that it actually carried the
strongest incremental signal in the whole feature set. Had the phase relied on correlation
alone, this feature would have been wrongly discarded. **Outcome**: a 5-feature core set that
survived every model-family comparison run afterward.

## Phase 3 — Preprocessing
**Objective**: build leakage-free, model-aware preprocessing. **Major decision**: don't just
assert leakage prevention matters — demonstrate its cost directly. **Evidence used**: a
leaky-vs-correct comparison across 30 reshufflings showed naive preprocessing is
optimistically biased in 29 of 30 trials. **Mistake / infrastructure failure**: long-running
Python jobs backgrounded by the environment were silently killed with no error twice during
this phase, forcing a switch to foreground execution and, for a few specific benchmarks, a
reduced CV budget (5×5 instead of 5×10) — a disclosed, not hidden, reduction in statistical
power for those comparisons. **A counter-intuitive finding trusted because it was measured,
not assumed**: KNN performed *worse* with every scaler tested than with none, contradicting
the standard "always scale for distance models" default. **Outcome**: a per-model-family
preprocessing configuration, not a one-size-fits-all pipeline.

## Phase 4 — Baseline Models
**Objective**: compare 13 model families under one identical, leakage-free protocol.
**Major decision**: use Friedman + Nemenyi correction, not raw leaderboard ranking, to decide
which models were *actually* separable from each other. **Evidence used**: Friedman
χ²=530.2, p=8.2e-106 (something is different); Nemenyi then showed ExtraTrees was **not**
statistically distinguishable from CatBoost (p=0.953) despite a significant uncorrected
paired test (p=1.5e-8) — the naive "ExtraTrees wins" conclusion was wrong. **Bug found**:
permutation importance computed on the same data a model was fit on gave meaningless,
inflated values for models with near-zero training error — traced to the baseline score
being ~0 (nothing to degrade from), fixed by evaluating on a held-out split. **What was
dropped, and why**: NGBoost failed on a confirmed `sklearn`/`ngboost` version incompatibility;
EBM never completed a single benchmark pass within the runtime budget. Both are disclosed as
engineering limitations, not performance judgments — we don't know how either would have
scored. **Outcome**: a top statistical tier of 4 models promoted to tuning, not just 1.

## Phase 5 — Validation Strategy & Hyperparameter Optimization
**Objective**: pick a validation protocol with evidence, then tune only the 4 shortlisted
models. **Major decision**: directly compare RepeatedKFold, Nested CV, Monte Carlo CV, and
Bootstrap on measured criteria (reseed-stability, bias, cost) rather than picking one by
convention. **Counter-intuitive finding, investigated not dismissed**: nested CV's "naive"
estimate was *worse*, not better, than its honest outer estimate — the opposite of the
textbook optimistic-bias story. Traced to smaller inner-fold training sets interacting with
this model's steep learning curve at small n, not to an absence of the classic bias. **The
single most important event of the whole project**: hyperparameter tuning made ExtraTrees —
the best baseline model — **statistically significantly worse** (p=0.046), caught only
because every Optuna-selected configuration was mandatorily re-validated under the full
protocol before being trusted. The recommendation followed the evidence (keep ExtraTrees at
defaults) rather than the tuning process's own output. **Outcome**: 3 of 4 models genuinely
improved by tuning; the 4th was correctly reverted.

## Phase 6 — Ensemble Evaluation
**Objective**: determine, with evidence, whether ensembling could beat the single best model.
**Major decision**: don't let prediction correlation (≥0.92 among all 4 tuned models) settle
the question — measure disagreement rate and region-specific disagreement instead. **What
was found and NOT immediately trusted**: a linear blend produced a striking ~12% RMSE
improvement, but its fitted coefficients included a *negative* weight on RandomForest — a
pattern resembling a known collinearity-instability failure mode already seen with Ridge in
Phase 4. **How it was verified before being reported**: refit the meta-model independently on
each of 10 available CV repeats (same qualitative pattern every time), then a full
leave-one-repeat-out nested validation. **A genuine, not-just-cosmetic finding**: clipping
predictions to [0,100] *improved* accuracy, not just enforced plausibility — a negative
prediction for a true near-zero row becomes exactly correct once clipped. **Outcome**: a
3-model blend (GaussianProcess dropped, p=0.615 to remove it), RMSE 14.76 vs. 16.69,
p<0.000001, validated broadly across every physically meaningful operating region tested.

## Phase 7 — Submission Pipeline & Lockdown
**Objective**: freeze the model and build a deterministic, validated inference pipeline.
**Major decision**: test the pipeline adversarially, not just on the happy path. **Bug found
this way**: the submission validator crashed with a raw, unhandled exception on a malformed
header instead of failing cleanly — found by deliberately constructing bad inputs, fixed
immediately. **Outcome**: 3/3 identical SHA-256 hashes across independent runs, a git tag
(`submission_v1`), and a machine-readable manifest.

## Phase 8 — Technical Defense & Presentation Prep
**Objective**: prepare to defend every decision to a skeptical panel. **Major decision**:
write the red-team review as an outside reviewer trying to reject the project, not as its
author summarizing strengths. **What this surfaced**: the feature-selection-on-full-dataset
leakage risk (real, not fold-level, never fully corrected), and the ensemble's
in-distribution-only validation status. **A specific, evidence-gathered answer prepared in
advance**: rather than speculate about why 42% of test predictions were exactly zero, the
actual test-set feature values were checked — predicted-zero rows had a mean `avg_temp` of
467.6K, within 1.5K of the training zero-yield group's mean. **Outcome**: a full judge
question bank, confidence audit, and presentation materials, all evidence-cited.

## Phase 9 — Final Independent Audit
**Objective**: one more adversarial pass, explicitly framed as an outside Kaggle
Grandmaster/Senior Engineer auditing someone else's work. **What this found that no prior
phase caught**: the submission CSV displayed fewer than 3 decimal places on 25 of 50 rows
(pandas silently drops trailing zeros), a literal violation of the competition's stated
format requirement and a real risk of automated rejection — **fixed immediately**, with a new
validator check added so this exact class of bug cannot recur silently. Also computed the
project's first explicit `DummyRegressor` baseline (should have existed since Phase 4) and
quantified, for the first time with a concrete number, the project's multiple-comparisons
exposure (~300-500 exploratory tests, ~15-25 expected false positives at α=0.05) and named a
"winner's curse" risk from reporting the best of 22 logged experiments. **Outcome**: score
7.9/10, recommendation "ready after administrative action" (rename the placeholder team
name).

## Phase 10 — Project Verification Certificate
**Objective**: produce the authoritative record of the exact submission artifact.
**Major decision**: generate JSON, Markdown, and PDF from one assembled fact dictionary so
the three formats cannot disagree. **What this surfaced**: a PDF-rendering library
(`xhtml2pdf`) silently ignoring modern CSS table-layout properties, causing visual column
overlap — diagnosed by actually rendering the PDF to images and inspecting it, not assumed
correct from clean-looking HTML source; fixed with HTML4-style width attributes. Also
verified, by direct hash comparison, that git's line-ending handling did not silently corrupt
the binary PDF during staging. **Outcome**: a certificate whose every fact was computed or
read live in that session, not carried forward from memory.

---

# Part 2 — The Ten Highest-Leverage Decisions

1. **Refusing to start with modeling (Phase 0).** Every engineered feature in Phase 2 traces
   to a specific physical hypothesis stated *before* the data was examined. Without this,
   feature engineering would have been undirected correlation mining — slower, and far
   weaker material for defending the project to judges. *What could have happened otherwise*:
   a purely data-driven feature search would likely have found some of the same signals
   (e.g. `avg_temp`'s dominance is hard to miss) but would have had no principled way to
   distinguish `residence_proxy`'s genuinely non-monotonic signal from noise, since that
   required knowing to look for it via Mutual Information specifically.

2. **Treating the zero-yield mass as a question, not an assumption (Phase 1).** Verifying it
   with a gap analysis and GMM check, rather than eyeballing a histogram and moving on, gave
   the project a defensible, quantified claim instead of a vague impression — directly
   reused in Phase 2's classifier-separability check and Phase 8's zero-prediction
   investigation.

3. **Rejecting 19 of 24 engineered features (Phase 2).** A benchmark directly showed that
   dumping in all 24 candidates *hurt* every model tested relative to the curated 5-feature
   set. Without this discipline, the final model would likely have been both worse and
   harder to explain to judges.

4. **Discovering and proving preprocessing leakage bias empirically (Phase 3).** Rather than
   stating "we prevent leakage" as an assertion, the project measured the leaky alternative's
   actual bias (29/30 trials optimistic) — turning a design principle into a demonstrated
   fact, and the single cleanest slide in the technical defense.

5. **Choosing the validation protocol by comparison, not convention (Phase 5).**
   RepeatedKFold(5,10) wasn't picked because it's popular — it was picked after being
   measured against three alternatives on reseed-stability, bias, and cost, with the
   comparison's own counter-intuitive result (nested CV's naive estimate being pessimistic,
   not optimistic) investigated rather than glossed over.

6. **Rejecting tuned ExtraTrees despite having just spent real compute tuning it (Phase 5).**
   This is the project's strongest single piece of evidence that its own process works: the
   mandatory re-validation step caught a regression the tuning search itself never would have
   reported, and the final recommendation followed the evidence over the sunk cost.

7. **Not trusting the ensemble result until it survived 10 independent stress tests
   (Phase 6).** A ~12% RMSE improvement with a suspicious coefficient pattern was treated as
   guilty until proven innocent, not as a win to report immediately. The eventual leave-one-
   repeat-out validation is the single most rigorous statistical procedure in the entire
   project.

8. **Building deterministic, adversarially-tested inference (Phase 7).** Testing the
   pipeline against malformed inputs — not just confirming the happy path — found a real
   crash bug before it could reach a judge.

9. **Running a genuinely adversarial final audit instead of a second self-congratulatory
   review (Phase 9).** This is the phase that found the project's most severe remaining
   defect (the CSV decimal-format bug) — a defect that two prior "validation" passes
   (Phase 7, Phase 8) had not caught, because neither had re-read the competition's literal
   format requirement against the literal file bytes.

10. **Generating the verification certificate from one source of truth, then actually
    rendering and inspecting it (Phase 10).** Treating "the PDF looks right" as a claim to be
    checked, not assumed, caught a real rendering bug (column overlap) that would otherwise
    have shipped in the one document explicitly meant to be presentation-ready.

---

# Part 3 — Every Important Mistake, and What It Cost

**1. Quadratic-R² shape diagnostic mischaracterized `avg_temp`'s relationship as linear
(Phase 2).**
*What happened*: an automated diagnostic comparing linear vs. quadratic R² reported near-zero
quadratic gain, suggestive of "already linear."
*How discovered*: visually inspecting the actual LOWESS curve showed an obvious sigmoidal
shape a quadratic term is mathematically ill-suited to detect.
*What prevented worse consequences*: the correction happened within the same phase, before
any downstream model or claim depended on the wrong characterization.
*How to avoid in future*: never trust a single automated shape diagnostic without a
visual check — different diagnostics are sensitive to different shape families, and none
covers all of them.

**2. Permutation importance computed in-sample gave meaningless results (Phase 4).**
*What happened*: importances of 20-40 RMSE units were reported for models with near-zero
in-sample error, since the "baseline" score being degraded from was itself near-perfect and
therefore artificially fragile.
*How discovered*: the magnitudes were implausibly large relative to the models' actual CV
RMSE, prompting a re-check.
*What prevented worse consequences*: the numbers were sanity-checked against known CV
performance before being reported anywhere external.
*How to avoid*: always compute permutation/feature importance on a held-out split for any
model capable of memorizing its training data, as a default, not a special case.

**3. The submission validator crashed on malformed input instead of failing cleanly
(Phase 7).**
*What happened*: `validate_submission_file` unconditionally indexed a column that an earlier
check had already found missing, raising a raw `KeyError`.
*How discovered*: deliberately constructing a malformed test submission as part of
adversarial testing.
*What prevented worse consequences*: this was caught during internal testing, not during an
actual judge's evaluation.
*How to avoid*: guard every "downstream" check behind the specific condition an earlier
check already established, rather than assuming a linear pass-through.

**4. Submission CSV displayed fewer than 3 decimal places on half its rows (Phase 9,
discovered two phases late).**
*What happened*: `np.round(x, 3)` correctly rounded the underlying float, but pandas' default
CSV writer drops trailing zeros on write, so `0.000` became `0.0`.
*How discovered*: a Phase 9 adversarial audit re-read the competition's literal format
requirement and checked the raw file bytes, not just the parsed DataFrame.
*What prevented worse consequences*: this was found and fixed before the actual competition
upload — but it existed, undetected, through both Phase 7's "submission validated" checkpoint
and Phase 8's full technical defense. Neither prior phase re-verified the literal string
output against the literal spec; both trusted that "the value rounds correctly" was
sufficient.
*How to avoid*: when a specification says "formatted as X," test the literal output bytes
against that requirement — never infer compliance from the correctness of the underlying
numeric operation.

**5. No trivial `DummyRegressor` baseline existed until Phase 9.**
*What happened*: 13 real models were benchmarked, tuned, and ensembled — and none of that
was ever compared against the simplest possible baseline (predicting the training mean).
*How discovered*: a Phase 9 audit, framed explicitly as "what would a Kaggle Grandmaster ask
for first," flagged the omission.
*What prevented worse consequences*: nothing — this was a documentation/rigor gap, not a
correctness bug; the real models were never at risk of losing to a dummy baseline, we simply
had no number on record proving it.
*How to avoid*: benchmark a dummy baseline in the *same script*, in the *same phase*, as the
first real model comparison — not as an afterthought.

**6. Long-running background processes were silently killed multiple times (Phases 3-5).**
*What happened*: several Optuna studies and CV benchmarks, when run as backgrounded shell
processes, terminated with no error message and no partial output, more than once.
*How discovered*: expected output simply never arrived; investigated by checking for a live
process and finding none.
*What prevented worse consequences*: Optuna's SQLite-backed study storage (adopted
specifically because of this recurring failure) meant no completed trial was ever lost —
studies resumed exactly where they left off. Reduced CV budgets for a few Phase 3 benchmarks
were a smaller, disclosed cost.
*How to avoid*: any long-running computation in an environment with unreliable background
execution should checkpoint incrementally by design, before the first failure, not after.

**7. Feature selection was never nested inside cross-validation.**
*What happened*: which of 24 candidate features to keep was decided using statistics computed
on the full training set, not re-derived per outer fold.
*How discovered*: named directly during the Phase 8 red-team review and Phase 9 audit, not
"discovered" via a symptom — this is a design-time gap identified through deliberate
self-scrutiny, not an observed failure.
*What prevented worse consequences*: the promoted features were each required to pass
multiple independent tests (an informal, more conservative bar than any single p<0.05),
likely limiting how much this actually inflated reported performance — but the magnitude was
never directly measured.
*How to avoid*: for any project small enough that this is computationally feasible, nest
feature selection inside the outer validation loop from the start, rather than treating it
as a one-time, whole-dataset decision.

**8. No formal project-wide multiple-comparisons correction was ever applied.**
*What happened*: an estimated 300-500 statistical tests were run across the exploratory
phases without a Bonferroni/FDR-style correction across the whole project (individual
phases, like the 13-model Nemenyi comparison, did correct within their own scope).
*How discovered*: quantified for the first time during the Phase 9 audit by roughly counting
tests run per phase.
*What prevented worse consequences*: the headline claims that actually drove the final model
(the ensemble's improvement, p<0.000001) have effect sizes large enough to survive almost any
reasonable correction; the exposure is concentrated in the many *rejected* exploratory
findings, which were never acted on anyway.
*How to avoid*: decide on a multiple-testing strategy (per-phase correction vs. project-wide)
before the exploratory phase begins, not after noticing the exposure in hindsight.

---

# Part 9 — Final Reflection

This project did not teach us that our model is good. It taught us something more durable:
**almost every genuinely useful finding in this project came from refusing to trust a first
result** — the zero-yield gap, the feature that looked redundant but wasn't, the tuning that
looked like an improvement but wasn't, the ensemble coefficient that looked like a red flag
and turned out to be real, and the submission format that looked compliant and wasn't. None
of these were caught by being smart in the moment they were first produced. They were caught
by a second pass that assumed the first pass might be wrong.

Our understanding evolved in a specific direction: from "validate the model" toward "validate
the validation." By Phase 5, checking a hyperparameter search's own output before trusting it
was routine. By Phase 9, checking a *prior audit's* own coverage (did Phase 7/8's validation
actually re-check the literal spec, or just the parsed data?) was what caught the most severe
remaining defect. Each layer of scrutiny found something the layer below it had missed — not
because any single layer was careless, but because verification has to be adversarial and
literal to catch certain classes of error, and no single pass reliably is both.

The assumption that turned out most wrong was an implicit one, never stated outright until it
was tested: that "the tuning process found a better configuration" and "re-validating that
configuration under a stricter protocol will confirm it" were the same claim. They were not,
for ExtraTrees, and the gap between them was the most informative result the project
produced about its own methodology.

What proved indispensable was not any specific algorithm or feature — it was the discipline
of writing down, at every phase, the evidence for a claim *before* trusting the claim, and
being willing to let that evidence overturn a prior phase's conclusion (Phase 2's SHAP
disagreement understood as an artifact, Phase 5's tuning reverted, Phase 9's format bug
fixed against the freeze). A project that had instead treated each phase's conclusion as
final would have shipped a worse model with more unearned confidence in it.

What should never be forgotten: **a validation step that has not itself been tested against
a case where it should fail is not yet a validation step — it is an assumption wearing a
validation step's clothes.** The CSV decimal-place bug survived two "validated" checkpoints
specifically because neither one had been tested against a submission that was subtly wrong
in that particular way. The fix was not more validation; it was validation of the validation.
