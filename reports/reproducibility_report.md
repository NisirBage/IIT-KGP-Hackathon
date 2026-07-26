# Phase 7 — Reproducibility Report

## 1. Repeated-run determinism (Core Principle 3: every prediction must be reproducible)

The inference pipeline was run **3 independent times** in separate Python processes, from a
cold state each time (fresh interpreter, fresh model deserialization, fresh CSV read):

| Run | Output SHA-256 |
|---|---|
| 1 | `547521f4b9a249a6650927164433ca6d5e27f5a8684297fa69897a499d1c1c94` |
| 2 | `547521f4b9a249a6650927164433ca6d5e27f5a8684297fa69897a499d1c1c94` |
| 3 | `547521f4b9a249a6650927164433ca6d5e27f5a8684297fa69897a499d1c1c94` |

**Identical byte-for-byte output across all 3 runs.** This is expected and by design: every
base model in the ensemble (`ExtraTrees`, `CatBoost`, `RandomForest`) has a fixed
`random_state=42` baked into its serialized, already-fitted state — inference involves no
retraining, no resampling, and no randomness of any kind, only deterministic `.predict()`
calls through a fixed pipeline.

## 2. Input-validation fail-loud testing

Four categories of malformed input were constructed and passed through
`validator.validate_input_schema()` to confirm the pipeline fails loudly (raises
`ValidationError` with a specific, actionable message) rather than silently producing a
wrong prediction:

| Malformed input | Result |
|---|---|
| Missing a required column (`jacket_temperature_K`) | Raised: `"missing required columns ['jacket_temperature_K']"` |
| Wrong row count (10 instead of 50) | Raised: `"expected 50 rows, got 10"` |
| A missing (NaN) value in one cell | Raised: `"1 missing values in input data"` |
| An unexpected extra column | Raised: `"unexpected columns ['extra_col']"` |

All four passed as designed.

## 3. Submission-format fail-loud testing (and a bug this caught)

Two malformed submission files were constructed and passed through
`validator.validate_submission_file()`:

| Malformed submission | Result |
|---|---|
| Wrong header (`yield` instead of `overall_yield`) | Initially crashed with an unhandled `KeyError` — **bug found and fixed** (see `submission_pipeline_report.md`); after the fix, raised a clean `ValidationError: ['header_name', 'float_values']` |
| Written with `index=True` (extra unnamed index column) | Raised: `ValidationError: ['column_count', 'header_name', 'no_index_column']` |

The real submission file (`submission/TeamName.csv`) was re-validated after the fix and
still passes cleanly (`validate_submission_file` → `passed: True`).

## 4. Serialization round-trip

The final ensemble was `joblib.dump`'d and `joblib.load`'d back, and predictions on the full
training set were compared before and after: **identical** (`np.allclose` = True), matching
the equivalent check already performed for the standalone pipelines in Phase 3
(`preprocessing_report.md`) and Phase 5 (`final_model_selection_report.md`).

## 5. Sources of nondeterminism explicitly ruled out

| Potential source | Status |
|---|---|
| Random forest / extra trees bootstrap sampling | Fixed at training time (`random_state=42`), frozen into the serialized artifact — not re-sampled at inference |
| CatBoost's internal randomization (`random_strength`, `bagging_temperature`) | Same — fixed at training time, not re-invoked at inference |
| Floating-point summation order across threads | Single-threaded inference (`n_jobs` not set to parallel anywhere in the inference path); not observed to vary across the 3 runs above |
| Pandas/numpy version-dependent hashing or iteration order | Controlled by `requirements_frozen.txt` pinning exact versions; not tested across a *different* environment (see `competition_readiness_report.md` risk assessment) |

## Conclusion

All reproducibility checks pass. The pipeline is deterministic across repeated runs in the
frozen environment, and the validator has been adversarially tested (not just tested on the
happy path) against both malformed inputs and malformed outputs — with one real bug found
and fixed as a direct result of that adversarial testing.
