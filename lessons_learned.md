# Lessons Learned

Distilled from [`project_retrospective.md`](project_retrospective.md) — this document is the
"so what do we actually change" companion, not a restatement of what happened.

---

# Part 4 — Engineering Principles That Emerged

Each principle below is included only because a specific event in this project demonstrated
it — not as generic advice.

**1. Never jump to modeling before understanding the domain.**
*Demonstrated by*: Phase 0's series-reaction reasoning directly predicted the non-monotonic
residence-time signature that Phase 2 later confirmed via Mutual Information — a signal that
would have been very easy to dismiss as noise (near-zero linear correlation) without knowing
in advance to look for a non-monotonic relationship specifically.

**2. Every hypothesis deserves an experiment, not an eyeball check.**
*Demonstrated by*: the zero-yield mass (Phase 1) was tested with a gap analysis and a GMM
component comparison rather than accepted from a histogram; the avg_temp shape (Phase 2) was
corrected only because a LOWESS curve was actually plotted and inspected, not inferred from a
single R² number.

**3. Evidence beats intuition — including your own project's prior intuition.**
*Demonstrated by*: Phase 5's decision to keep ExtraTrees at its defaults directly
contradicted the intuitive expectation that "the model we just spent compute tuning must be
at least as good." The evidence (a statistically significant regression under the full
validation protocol) won.

**4. Simplicity wins unless complexity proves itself.**
*Demonstrated twice*: Phase 2's feature benchmark showed 24 features performing worse than a
curated 5; Phase 6's ensemble evaluation showed a 4-model blend performing statistically
identically to a 3-model one, so the 4th model (GaussianProcess) was dropped despite having
been a legitimate, evidence-based shortlist member.

**5. A counter-intuitive result is a reason to investigate harder, not a reason to discard
it or accept it uncritically.**
*Demonstrated by*: the ensemble's negative RandomForest coefficient (Phase 6) was neither
thrown out (because it looked like a known failure pattern) nor reported immediately (because
the RMSE improvement was real) — it was stress-tested 10 independent ways specifically
because it was surprising in both directions.

**6. Every validation rule should itself be validated — ideally by trying to break it.**
*Demonstrated by*: the Phase 7 validator crash and the Phase 9 decimal-format bug were both
found by deliberately constructing inputs designed to expose a weakness, not by running the
pipeline normally and observing that it seemed to work.

**7. Reproducibility is a feature you build and test, not a property you assume.**
*Demonstrated by*: determinism was verified by actually running the pipeline multiple times
and comparing hashes (Phase 7, re-confirmed live in Phase 9 and Phase 10) — not asserted from
"we used fixed random seeds."

**8. Documentation is part of engineering, and its absence is itself a defect.**
*Demonstrated by*: the missing `DummyRegressor` baseline (not caught until Phase 9) was not a
correctness bug — the real models were never at risk — but it was a genuine gap in the
project's ability to defend itself, treated with the same seriousness as a code bug once
found.

**9. The most dangerous errors are the ones that pass a shallow check.**
*Demonstrated by*: the CSV decimal-place bug passed dtype checks, row-count checks, and
header checks — every check that looked at the *parsed* data. It only failed a check that
looked at the *literal file bytes*, because pandas silently reformats "0.0" back into a
perfectly valid float on read, hiding the very discrepancy that mattered.

**10. A second independent audit finds different things than the first, even when both are
done in good faith.**
*Demonstrated by*: Phase 9's audit found the CSV bug and the missing dummy baseline — neither
of which Phase 7 (submission validation) or Phase 8 (technical defense, itself a red-team
exercise) had caught. Good-faith self-review has a ceiling; a fresh adversarial pass run with
a different specific question in mind ("would this survive a literal spec re-check?") finds
different defects than a fresh pass run with a different one ("would this survive a
skeptical judge's questions?").

---

# Part 6 — What Would We Do Differently, Starting From Zero Tomorrow

**What would remain identical**:
- The phase ordering (physics → EDA → features → preprocessing → models → tuning →
  ensemble → freeze → defense → audit → certificate). Every phase's output was a real,
  necessary input to the next; none felt like ceremony in hindsight.
- The insistence on statistical evidence for every promoted claim, and the discipline of
  writing a report before moving to the next phase rather than after the whole project.
- The decision to build reusable `src/` modules (features, preprocessing, models,
  optimization, inference) rather than one-off notebook scripts per phase.

**What would change**:
- **A trivial baseline (`DummyRegressor`) would be benchmarked in the very first modeling
  script**, alongside the first real model, not left until an audit noticed its absence.
- **Feature selection would be nested inside cross-validation from the start**, even at the
  cost of more compute, rather than accepted as a disclosed-but-unresolved limitation.
- **The submission format specification would be re-read literally, against literal output
  bytes, the moment a submission file first exists** — not treated as satisfied once the
  underlying numeric values are correct. This single change would have caught the Phase 9
  bug two phases earlier.
- **Checkpointing for any long-running computation would be built in before the first
  background-execution failure**, not adopted reactively after losing progress twice.

**What would be skipped**:
- Nothing substantive. Even the two "optional" models that never got benchmarked (NGBoost,
  EBM) aren't things we'd skip trying — we'd still attempt them, just with more schedule
  buffer for the specific failure modes encountered (a version-incompatibility issue and a
  runtime budget issue respectively).

**What would be added**:
- An adversarial-testing checklist applied at the *end of every phase that produces a
  user-facing artifact* (not just at the final freeze) — malformed-input testing,
  literal-spec re-verification, and a "what would a hostile reviewer check" pass, run
  incrementally rather than concentrated into two late audit phases.
- A running, explicit tally of "how many hypothesis tests have we run so far," updated per
  phase, so the multiple-comparisons exposure is visible throughout rather than reconstructed
  after the fact in a single audit.

**What order would change**:
- The dummy-baseline computation and the literal submission-format check would both move from
  "wherever they happened to get caught" to fixed, mandatory checkpoints immediately after
  Phase 4 (first models) and immediately after Phase 7 (first submission file) respectively.

---

# Part 7 — Advice to Future Teams Attempting a Similar Competition

**Technical advice**:
- If your target has a hard physical bound (like `[0,100]`), test whether clipping to that
  bound *improves* your metric, not just whether it's required for compliance — in this
  project it did, meaningfully, because a negative prediction for a near-zero true value
  becomes exactly correct once clipped rather than merely closer.
- Don't trust prediction correlation alone to decide whether ensembling is worth attempting.
  Check disagreement rate and *where* models disagree (which operating regions) — a
  correlation of 0.95 can still hide a real, exploitable, region-specific disagreement.
- If your best model gets worse after hyperparameter tuning, that is not necessarily a sign
  your tuning process is broken — re-validate under your most rigorous protocol before
  concluding either way.

**Statistical advice**:
- Budget for a formal validation-protocol comparison (RepeatedKFold vs. Nested vs. Monte
  Carlo vs. Bootstrap) *before* committing to one, especially at small sample sizes — the
  "obviously correct" choice (nested CV, in the classic telling) can behave counter-intuitively
  at small n, and you want to find that out on a tuning exercise, not in a judge's Q&A.
- Track your total hypothesis-test count as you go, not retrospectively. A rough number
  ("we ran approximately N tests across the exploratory phase") is far better than no number,
  and takes almost no extra effort if tallied incrementally.
- If you find a surprising result (a negative ensemble coefficient, an unexpected sign
  reversal), the standard for accepting it should scale with how surprising it is, not just
  with whether it clears p<0.05 once.

**Software engineering advice**:
- If your execution environment has ever silently killed a long-running background job,
  assume it will happen again and design for resumability (checkpoint to disk incrementally)
  before you need it, not after losing the first run.
- Test your submission validator against inputs it should *reject*, not just inputs it should
  accept. A validator that has never seen a bad input is unverified, not passing.
- Re-read your competition's format specification against the literal bytes of your output
  file, not against your program's internal representation of that output — the two can
  silently diverge (as they did here) even when every unit test passes.

**Competition strategy**:
- Shortlist more than one "best" model if your statistical comparison doesn't cleanly
  separate the top few — tuning and ensembling both benefit from having real alternatives on
  the bench, and in this project the eventual winner wasn't the single best baseline model,
  it was a combination that only made sense once multiple strong candidates existed.
- Treat "only one submission allowed" as a reason to build and rehearse your entire
  submission pipeline well before the deadline, including adversarial testing — not a reason
  to rush the final steps.

**Presentation strategy**:
- Prepare your most counter-intuitive or risky-looking finding as a dedicated slide with its
  own evidence, before a judge asks about it — in this project, that meant a specific,
  data-backed slide on why 42% of test predictions were exactly zero, ready in advance rather
  than improvised.
- State your genuine limitations before being asked. A panel that hears "here's what we're
  not sure about, and why" reads as more credible than a panel that has to extract it via
  cross-examination — and every limitation named in this project's technical defense was one
  we'd already found ourselves, not one a judge discovered first.
