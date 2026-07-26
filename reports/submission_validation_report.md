# Phase 7 — Submission Validation Report

Full machine-readable trace: [`submission/last_inference_report.json`](../submission/last_inference_report.json).
This report is the human-readable summary of that trace, run against the real
`data/raw/test_dataset.csv`.

## Input validation (11 checks, all passed)

| Check | Result |
|---|---|
| No missing columns | Pass |
| No unexpected columns | Pass |
| Column order matches expected | Pass |
| All 5 columns numeric (float64) | Pass |
| No missing values | Pass (0 found) |
| No duplicate rows | Pass (0 found) |
| Row count = 50 | Pass |

## Prediction validation (5 checks, all passed)

| Check | Result |
|---|---|
| Prediction count = 50 | Pass |
| No NaN values | Pass (0 found) |
| No infinite values | Pass (0 found) |
| Within [0,100] physical bounds | Pass — range [0.0000, 92.3464] |
| Duplicate-prediction flag (informational) | 30/50 unique values (not fatal — see note below) |

**Prediction summary statistics**: mean=29.198, std=32.329, median=14.859, min=0.000,
max=92.346. **21 of 50 test-set predictions (42%) are exactly at the lower clip bound
(0.0)** — the explicit `n_at_lower_bound` count from the validator. This is a real,
noteworthy characteristic of the output, not a validation failure, and is documented rather
than smoothed over:

- The training set's zero-yield rate is 24.7% (37/150) — the test set's *predicted*
  zero-rate (42%) is meaningfully higher. Two explanations are consistent with everything
  established earlier in this project: (a) the test set's operating conditions may sample
  the high-jacket-temperature / long-residence "collapse" region more heavily than training
  did (a distribution-shift question Phase 1's PSI check already flagged as a `moderate`,
  not resolved, concern for `length_m` specifically), or (b) the ensemble — built from models
  independently confirmed to be more accurate in the collapsed regime than the active one
  (`residual_analysis_report.md`, Phase 4) — could be systematically over-predicting
  collapse when genuinely uncertain, since the clip floor is a "safe" (physically valid) place
  to land when a component prediction goes negative. Neither explanation is verifiable
  without true test-set labels (which this project never has access to, by design — the
  labels are held by the competition). **This is worth stating proactively in the Phase 8
  presentation rather than waiting for a judge to ask why so many test predictions are
  exactly zero.**
- 0 predictions hit the *upper* bound (100) — no evidence of saturation at the high end.

## Submission file validation (6 checks, all passed)

| Check | Result |
|---|---|
| UTF-8 encoding | Pass |
| Row count = 50 | Pass |
| Column count = 1 | Pass |
| Header = `overall_yield` exactly | Pass |
| No index column | Pass |
| Float/numeric values | Pass (float64) |

## Clipping statistics

`n_values_changed_by_clip = 0` — the `LinearBlendEnsemble`'s internal clip already handled
every out-of-range value; the inference pipeline's redundant external clip
(`submission_pipeline_report.md`) changed nothing further on this run, confirming the two
clipping layers are consistent with each other, not compensating for a bug in either.

## Final artifact

| | |
|---|---|
| Path | `submission/TeamName.csv` |
| SHA-256 | `547521f4b9a249a6650927164433ca6d5e27f5a8684297fa69897a499d1c1c94` |
| Rows | 50 |
| Columns | 1 (`overall_yield`) |

**Note on filename**: `TeamName.csv` is a placeholder (`src/inference/config.py:TEAM_NAME`)
— replace with the actual competition team name before final upload; this is a one-line
config change, not a pipeline change, and does not require re-running inference (the CSV
content is identical regardless of filename).

## Conclusion

All 22 automated checks across input, prediction, and submission-file validation passed on
the real test dataset. The one informational flag (high zero-prediction rate) is not a
validation failure and has been investigated and documented rather than ignored.
