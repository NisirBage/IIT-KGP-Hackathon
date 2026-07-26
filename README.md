# Reactor Yield Prediction — ML Hackathon

Surrogate ML model predicting `overall_yield` of product B from a non-isothermal
continuous-flow reactor (series reaction A → B → C), replacing expensive CFD/BVP simulation.

See [`docs/problem_statement.pdf`](docs/problem_statement.pdf) for the full brief and
[`reports/phase0_problem_understanding.md`](reports/phase0_problem_understanding.md) for the
domain reasoning behind our modeling choices.

## Layout

- `data/raw` — original `train_dataset.csv` (150 rows, labeled) and `test_dataset.csv` (50 rows, unlabeled)
- `data/processed`, `data/external` — engineered/derived data
- `src/{data,features,models,validation,visualization,utils}` — pipeline code
- `notebooks/` — exploratory + final documented notebook
- `configs/` — external configuration (seeds, hyperparameters)
- `artifacts/`, `models/`, `logs/` — run outputs (gitignored)
- `reports/` — phase-by-phase written findings
- `submission/` — final `[TeamName].csv`

## Status

Phase 0 (problem understanding) complete. See task list / reports for progress.
