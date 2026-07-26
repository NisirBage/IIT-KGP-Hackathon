# Phase 7 — Submission Pipeline Report

Code: [`src/inference/`](../src/inference/) (`config.py`, `loader.py`, `validator.py`,
`pipeline.py`, `submission.py`, `predict.py`). Single command to generate the final
submission from raw test data, per Core Principle 5:

```bash
python -m inference.predict
```

(run from `src/`, or `python src/inference/predict.py` from the project root)

## Pipeline architecture

```
data/raw/test_dataset.csv
    │
    ▼
loader.load_test_data()          — reads raw CSV, no transformation
    │
    ▼
validator.validate_input_schema() — column names/order/dtypes/missing/duplicates/row count
    │                                (raises ValidationError and halts on any failure)
    ▼
loader.load_model()               — deserializes FINAL_ENSEMBLE_blend_v1.joblib
    │
    ▼
model.predict(df)                 — feature engineering + preprocessing + base-model
    │                                predictions + linear combination, ALL INSIDE the
    │                                loaded object (Core Principle 2: zero manual
    │                                preprocessing outside the pipeline)
    ▼
np.clip(preds, 0, 100)            — explicit, unconditional clip (redundant with the
    │                                ensemble's own internal clip, by design — see below)
    ▼
validator.validate_predictions()  — count/NaN/inf/range/summary stats
    │                                (raises ValidationError and halts on any failure)
    ▼
submission.write_submission()     — writes submission/TeamName.csv (UTF-8, no index,
    │                                single 'overall_yield' column, 3-decimal floats)
    ▼
validator.validate_submission_file() — re-reads the just-written CSV and re-checks
                                        every competition format requirement
```

## Why the pipeline clips twice

`LinearBlendEnsemble.predict()` (Phase 6) already clips internally. `pipeline.py` clips
*again*, unconditionally, immediately after calling `.predict()`. This is deliberate, not an
oversight: **the inference pipeline's correctness must not silently depend on an
implementation detail of one specific model class.** If a future model swap ever used an
ensemble class without internal clipping, the inference pipeline would still enforce the
physical bound. The redundant clip is a zero-cost safety net, confirmed on the real test set
to change 0 of 50 predictions (the internal clip already did its job) — see
`submission_validation_report.md`.

## What "identical to training" means here, concretely

The loaded artifact is a `LinearBlendEnsemble` (`src/models/ensemble.py`) whose 3 base
elements are full `sklearn.Pipeline` objects built by
`preprocessing.pipelines.build_pipeline(...)` — the exact same factory function used for
every benchmark since Phase 3. Calling `.predict()` on raw test data triggers, per base
model, in order: `FeatureSetSelector` (the 10-feature core set, Phase 2) →
model-family-specific scaler (`none` for ExtraTrees/CatBoost/RandomForest per Phase 3) →
the tuned or default model itself. There is no code path in `src/inference/` that computes
or applies any feature or scaling transform manually — every transform a judge's held-out
data would need is already inside the deserialized object.

## Single end-to-end run (this phase's dry run)

```
Submission written to: submission/TeamName.csv
SHA-256: 547521f4b9a249a6650927164433ca6d5e27f5a8684297fa69897a499d1c1c94
All validations passed: True
Prediction summary: mean=29.198 std=32.329 min=0.000 max=92.346
```

Full step-by-step trace: [`submission/last_inference_report.json`](../submission/last_inference_report.json).

## A bug found and fixed during this phase (in scope per the freeze exception)

`validator.validate_submission_file()` originally crashed with a raw, unhandled `KeyError`
(instead of a clean `ValidationError`) when tested against a malformed CSV with the wrong
header — because a later check unconditionally indexed `df[config.TARGET_COLUMN]` even when
an earlier check had already established that column didn't exist. Fixed by guarding that
check behind an explicit `if config.TARGET_COLUMN in df.columns` (see
`src/inference/validator.py`). This is exactly the "fail loudly, never silently" principle
being enforced on the validator itself — an unhandled exception with a confusing traceback is
its own kind of failure to fail loudly, and was caught by deliberately testing malformed
inputs (`reproducibility_report.md` §3), not by accident.
