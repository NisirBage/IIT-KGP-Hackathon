# Phase 7 — Artifact Manifest

Machine-readable version: [`manifest.json`](../manifest.json) (project root). Full hash
listing: [`phase7_artifact_hashes.json`](phase7_artifact_hashes.json). All hashes are SHA-256,
computed against the frozen `submission_v1` commit (`0515615e40c1ffca13914a97cb0258dc344080db`).

## Core submission artifacts

| Artifact | Purpose | Version | SHA-256 (first 16 chars) |
|---|---|---|---|
| `artifacts/tuned_pipelines/FINAL_ENSEMBLE_blend_v1.joblib` | **The deployed model** — 3-model linear blend, clipped to [0,100] | v1 (Phase 6) | `67d5b7f6237ddc6d...` |
| `submission/TeamName.csv` | Final competition submission (50 rows, `overall_yield` column) | v1 (Phase 7) | `547521f4b9a249a6...` |
| `manifest.json` | Machine-readable freeze record (commit, versions, seeds, model spec) | v1 (Phase 7) | see file |
| `notebooks/competition_notebook.ipynb` | Presentation-ready, fully executed narrative notebook | v1 (Phase 7) | `f5ff98d3ce3e58c5...` |

## Component pipelines (base models of the final ensemble)

| Artifact | Purpose | Version | SHA-256 (first 16) |
|---|---|---|---|
| `artifacts/tuned_pipelines/ExtraTrees_SELECTED_FINAL_v1.joblib` | ExtraTrees base pipeline (Phase 4 defaults — Phase 5 proved tuning hurt this model) | v1 | `b57235f543963a75...` |
| `artifacts/tuned_pipelines/CatBoost_tuned_v1.joblib` | CatBoost base pipeline (Phase 5 tuned) | v1 | `2ec8ec5aa5c1a003...` |
| `artifacts/tuned_pipelines/RandomForest_tuned_v1.joblib` | RandomForest base pipeline (Phase 5 tuned) | v1 | `e2f866b0006fcfc0...` |
| `artifacts/tuned_pipelines/GaussianProcess_tuned_v1.joblib` | GP pipeline — evaluated, ultimately **excluded** from the final blend (Phase 6, p=0.615 to drop) | v1 | `d29ee5090aba40f8...` |
| `artifacts/tuned_pipelines/ExtraTrees_tuned_v1.joblib` | ExtraTrees Optuna-tuned config — **not used** (statistically worse than defaults, kept for audit trail) | v1 | (see hashes file) |
| `artifacts/pipelines/ridge_core_standard_v1.joblib` | Ridge pipeline from Phase 3 serialization test — reference only, not part of final model | v1 | (see hashes file) |

## Configuration & registries

| Artifact | Purpose | Version |
|---|---|---|
| `src/preprocessing/config.py` | `MODEL_FAMILY_PREPROCESSING` — per-model scaler/feature-set assignment | Phase 3, unchanged since |
| `src/models/registry.py` | `MODEL_REGISTRY` — every model's factory + preprocessing settings | Phase 4, unchanged since |
| `src/optimization/search_spaces.py` | Optuna search space definitions | Phase 5, unchanged since |
| `src/models/ensemble.py` | `LinearBlendEnsemble` class definition | Phase 6, unchanged since |
| `src/inference/config.py` | Frozen inference configuration (expected schema, model path) | Phase 7 |
| `reports/feature_registry.md` | Every candidate feature's status (Validated/Rejected/Pending) | Phase 2, amended Phase 3 |
| `reports/preprocessing_registry.md` | Every preprocessing decision with evidence | Phase 3 |
| `reports/experiment_registry.md` | Every experiment run in the project, EXP-000 through EXP-020 | Phases 1-6 |

## Optuna study artifacts (full hyperparameter search history)

| Artifact | Purpose | Trials | SHA-256 (first 16) |
|---|---|---|---|
| `artifacts/optuna_studies/ExtraTrees.db` | Full Optuna trial history (SQLite) | 80 target, 51 complete + 29 pruned | `24f38a814980c437...` |
| `artifacts/optuna_studies/CatBoost.db` | Full Optuna trial history | 60 target, 34 complete + 26 pruned | `c5f321aeb2a7e90d...` |
| `artifacts/optuna_studies/RandomForest.db` | Full Optuna trial history | 80 target, 59 complete + 21 pruned | `25567193a33c6ae8...` |
| `artifacts/optuna_studies/GaussianProcess.db` | Full Optuna trial history | 25 target, 17 complete + 8 pruned | `486a9706c6df6528...` |

Each `.db` is resumable — `optuna.load_study(...)` against it reconstructs the complete
trial-by-trial search history for audit or extension.

## Data

| Artifact | Purpose | SHA-256 (first 16) |
|---|---|---|
| `data/raw/train_dataset.csv` | Original 150-row training set (unmodified since Phase 0) | `b8b499bebf0f0edf...` |
| `data/raw/test_dataset.csv` | Original 50-row test set (unmodified since Phase 0) | `2a0b9f57a5c0e8e1...` |

## Reports (28 markdown files across 7 phases)

Full phase-by-phase index: [`manifest.json`](../manifest.json) → `phase_reports_index`. Not
individually hashed here (they are prose deliverables, not executable/deployed artifacts) —
their content is fixed as of the `submission_v1` commit and retrievable via
`git show submission_v1:reports/<name>.md`.

## Reproducing this manifest

```bash
git checkout submission_v1
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <path>
```
