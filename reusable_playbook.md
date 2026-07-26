# Reusable Competition Framework

Abstracted from this project's actual workflow — every step below corresponds to a phase we
genuinely executed, not an idealized process we're retrofitting. Domain-specific content
(reactor chemistry) is intentionally excluded from this document; see
[`project_retrospective.md`](project_retrospective.md) for that.

---

# Part 5 — The Generalized Workflow

```
Phase 0: Domain Understanding
    ↓  (produces: written, falsifiable hypotheses about the target variable's drivers)
Phase 1: Data Audit
    ↓  (produces: verified data quality facts + hypothesis test results, not assumptions)
Phase 2: Feature Engineering
    ↓  (produces: a small, statistically-validated feature set + a registry of rejected candidates)
Phase 3: Preprocessing
    ↓  (produces: a leakage-free, model-family-aware preprocessing pipeline, proven not assumed)
Phase 4: Baseline Model Comparison
    ↓  (produces: a statistically-grounded shortlist, not a single "best" model by raw metric)
Phase 5: Validation Strategy + Tuning
    ↓  (produces: a chosen validation protocol with evidence, and tuned candidates re-validated
    ↓   under that protocol before any improvement claim is trusted)
Phase 6: Ensemble Evaluation (conditional)
    ↓  (produces: a go/no-go decision on combining models, backed by disagreement analysis
    ↓   and stress-testing of any surprising result, not by correlation alone)
Phase 7: Submission Pipeline & Freeze
    ↓  (produces: a deterministic, adversarially-tested inference pipeline + a versioned release)
Phase 8: Technical Defense
    ↓  (produces: a genuine red-team review + anticipated questions with evidence-backed answers)
Phase 9: Independent Audit
    ↓  (produces: a second, differently-framed adversarial pass — catches what Phase 8 didn't)
Phase 10: Verification Certificate
    ↓  (produces: one authoritative, fact-checked record of the exact submitted artifact)
[Presentation]
```

Each arrow is a hard dependency, not a suggestion — every phase in this project consumed a
specific, concrete output of the phase before it (a feature registry, a preprocessing
config, a shortlist, a validation protocol, a frozen artifact). Skipping a phase doesn't just
skip effort, it removes an input the next phase needs.

## Phase-by-phase, generalized (works for Kaggle, industrial forecasting, classification,
scientific ML)

**Phase 0 — Domain Understanding.** Write down what you believe the target variable's
drivers are and why, in falsifiable form, *before* opening the dataset. For a classification
problem this might be "class separability should increase near decision boundary X because
of mechanism Y." The output is a hypothesis list, not a model.

**Phase 1 — Data Audit.** Test Phase 0's hypotheses against the actual data. Check for
target leakage, distribution shift between train/test, and any structural anomaly (bimodal
targets, unexpected zero-inflation, duplicate rows) with an actual statistical test — not a
glance at `.describe()`. Produce a dataset passport: a permanent record of what you verified
and how.

**Phase 2 — Feature Engineering.** Generate candidates motivated by Phase 0's hypotheses,
not by exhaustive automated feature generation. Validate each with more than one method
(correlation batteries alone will miss non-monotonic relationships; nested significance tests
alone will miss redundancy; redundancy checks alone will miss genuine multiplicative
interactions). Maintain a registry recording every candidate's status (validated/rejected/
pending) and why — this becomes the single reference for "did we already try this."

**Phase 3 — Preprocessing.** Build one pipeline object per model family, not one global
pipeline — different model families (bagged trees, boosting, distance-based, kernel-based)
genuinely have different preprocessing needs, and assuming otherwise costs real performance.
Prove leakage prevention works (measure a naive alternative's bias directly) rather than
asserting it.

**Phase 4 — Baseline Model Comparison.** Compare many model families under one identical
protocol (same folds, same seed, same metrics) so paired statistical tests are valid. Always
include a trivial baseline (predict the mean/mode/majority class) in this same comparison,
from the start — don't let it become a retrofit. Use a corrected multiple-comparison test
(Friedman/Nemenyi or equivalent) to decide which differences are real, not raw metric
ranking.

**Phase 5 — Validation Strategy + Tuning.** Choose your validation protocol by comparing
alternatives on measurable criteria (bias, variance/reseed-stability, computational cost) —
don't default to whatever's conventional without checking it fits your specific sample size
and data structure. When tuning, use a lighter search-time budget if needed for efficiency,
but *always* re-validate the winning configuration under your full, final protocol before
trusting any improvement claim — this is the single highest-value safeguard in the entire
framework.

**Phase 6 — Ensemble Evaluation (only if Phase 4-5 leave more than one strong candidate).**
Measure model disagreement directly (not just prediction correlation) before deciding
ensembling is or isn't worth attempting. Any surprising result from a fitted combination
(unusual coefficient signs, unstable weights) needs independent stress-testing — refit on
different data subsets and check the pattern holds — before being trusted, not before being
reported.

**Phase 7 — Submission Pipeline & Freeze.** Build one deterministic entry point from raw
input to final output. Validate it adversarially: construct inputs and outputs that *should*
fail, and confirm they do, cleanly. Freeze with a version tag and a machine-readable manifest
of the exact environment (versions, seeds, commit hash).

**Phase 8 — Technical Defense.** Write the review as if by someone trying to reject the
project, covering every phase for assumptions, weak evidence, and unjustified complexity.
Prepare specific answers, evidence-cited, for the hardest anticipated questions — especially
your own most surprising or riskiest finding.

**Phase 9 — Independent Audit.** A second adversarial pass, deliberately framed differently
from Phase 8 (e.g., "would this survive a literal specification re-check" rather than "would
this survive a Q&A") — different framings catch different defects. Re-verify compliance
against the literal submission specification, not your internal representation of it. Compute
any baseline comparison you haven't already (dummy baseline, naive heuristic) if missing.

**Phase 10 — Verification Certificate.** Generate one machine-readable fact record (commit
hash, environment, model configuration, performance, artifact hashes) and derive every other
format (human-readable document, presentation-ready document) from that single source, so
they cannot disagree. Verify every fact live at generation time rather than copying from
memory of earlier phases.

---

# Part 8 — Knowledge Worth Reusing As Templates

**The feature registry pattern** (`reports/feature_registry.md` in this project). A living
table of every candidate feature with formula, physical/domain motivation, statistical
support, and a status (Validated/Rejected/Pending). *Why reusable*: it prevents re-deriving
"did we already try this" from scratch in a later phase or a later project, and it's the
single artifact that makes a feature-engineering process auditable after the fact.

**The experiment registry pattern** (`reports/experiment_registry.md`). One row per
experiment: objective, configuration, exact validation protocol, result, and — critically —
"next action." *Why reusable*: it's what let this project answer "why did we choose X over
Y" months of work later without reconstructing the reasoning from memory, and it directly
enabled the Phase 9 "winner's curse" finding (counting how many experiments contributed to
the final reported number required only reading this file).

**The preprocessing factory pattern** (`src/preprocessing/pipelines.py`,
`src/preprocessing/config.py` in this project). A single function that builds a full
leakage-safe pipeline from a model-family name and a config dictionary, rather than
hand-writing a new pipeline per model. *Why reusable*: it's what made per-model-family
preprocessing (Phase 3's real finding) cheap to implement and test, instead of a maintenance
burden that would have discouraged the more accurate, model-specific choice.

**The two-tier tune-then-revalidate pattern** (`src/optimization/` + the mandatory final
re-validation step in Phase 5). Search with a lighter, faster validation budget; always
re-check the winner with your full, expensive, final protocol before trusting it. *Why
reusable*: this pattern is what caught the ExtraTrees regression — the single most important
finding in the project — and it generalizes to any optimization process (hyperparameters,
architecture search, feature selection) where the search budget and the trust budget are
different.

**The adversarial-audit workflow** (Phase 8 + Phase 9's structure: red-team review → judge
question bank → evidence-backed defense → a *second*, differently-framed adversarial pass).
*Why reusable*: a single self-review has a ceiling; running two adversarial passes with
different explicit framings (one for "will a domain/ML/stats expert challenge this," one for
"does this literally comply with the stated specification") found genuinely different
defects in this project. This is cheap to replicate and expensive to skip.

**The verification-certificate pattern** (`src/build_verification_certificate.py`,
`src/render_certificate.py`, `src/render_certificate_pdf.py` — one JSON fact-assembly step,
two rendering steps). *Why reusable*: generating multiple output formats from one source
of truth, with every fact computed live rather than copied, is a small amount of extra
engineering that eliminates an entire class of "the PDF says something different from the
JSON" inconsistency risk — worth the setup cost on any project with more than one archival
document to produce.

**The presentation-defense framework** (`judge_question_bank.md` +
`technical_defense.md` pairing: generate the hardest questions first, independently of the
answers, then answer each with a specific citation). *Why reusable*: writing the question
bank before drafting answers forces genuine coverage of weak points, rather than answers that
were pre-selected to sound good — the discipline is in the ordering, not just the content.

**What is *not* being recommended as a template**: the specific chemistry-derived feature
formulas (`avg_temp`, `residence_proxy`, etc.) are domain-specific to this reactor and should
not be copied into an unrelated project — only the *process* that produced them (hypothesis
first, then multi-method validation) generalizes.
