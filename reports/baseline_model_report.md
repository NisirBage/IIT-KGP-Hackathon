# Phase 4 — Baseline Model Report

Code: [`src/models/`](../src/models/) (`registry.py`, `pipelines.py`, `configs.py`, `benchmark.py`,
`metrics.py`, `evaluation.py`, `diagnostics.py`, `base.py`). Raw per-model checkpoints:
[`reports/phase4_raw/`](phase4_raw/) (one JSON per model — 50 fold-level metric rows +
out-of-fold prediction matrix each). Aggregate results:
[`phase4_analysis_results.json`](phase4_analysis_results.json).

**A reliability note, consistent with Phase 3**: long-running Python jobs backgrounded by
this environment were twice silently killed mid-run with no error or traceback (once during
the main benchmark loop, once during the EBM checkpoint). The benchmark script
(`run_phase4_benchmark.py`) was therefore designed to checkpoint each model to disk
immediately after it finishes and to skip already-checkpointed models on re-run — so the
only casualty of the interruptions was the EBM run itself (never produced a checkpoint,
treated as computationally infeasible within budget, see §4).

## 1. Framework

`models.registry.MODEL_REGISTRY` is the single place every candidate model is instantiated,
paired with its preprocessing settings (reusing Phase 3's `preprocessing.build_pipeline`).
`models.benchmark.run_repeated_cv` runs one manual `RepeatedKFold(5, 10)` loop per model —
identical seed (42) and identical fold sequence for every model, which is what makes the
paired statistical tests in `model_comparison_report.md` valid. Each pass through the loop
records, without any redundant refitting: per-fold RMSE/MAE/MedAE/R², fit/predict timing, and
a `(10, 150)` out-of-fold prediction matrix (one full 5-fold partition per repeat — every
sample gets exactly 10 independent held-out predictions, giving both a clean point estimate
for residual diagnostics and a genuine per-sample "prediction stability across folds" measure
that `sklearn.cross_val_predict` cannot produce for a *repeated* CV scheme).

## 2. Candidate models (13 run successfully, 2 optional excluded — see §4)

| Family | Models |
|---|---|
| Linear | LinearRegression, Ridge, Lasso, ElasticNet |
| Kernel | SVR (RBF), GaussianProcess |
| Instance-based | KNN |
| Tree Ensemble | RandomForest, ExtraTrees, HistGradientBoosting |
| Boosting | XGBoost, LightGBM, CatBoost |

Every model uses its preprocessing exactly as recommended in Phase 3
(`preprocessing.config.MODEL_FAMILY_PREPROCESSING`), extended with a documented
extrapolation for the models Phase 3 didn't directly benchmark (linear-family models get
Ridge's `power_yeojohnson` scaler; additional tree/boosting models get "no scaling",
extrapolating the scale-invariance Phase 3 empirically confirmed for RandomForest/CatBoost —
see `models/registry.py` docstring). No hyperparameter tuning anywhere — library defaults
throughout, exactly per the phase's guiding principle.

## 3. Leaderboard

| Rank | Model | Family | RMSE | Std | MAE | MedAE | R² | Fit time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | **ExtraTrees** | Tree Ensemble | **16.69** | 2.24 | 11.77 | 8.11 | **0.798** | 0.46 |
| 2 | **CatBoost** | Boosting | 17.99 | 2.48 | 12.86 | 8.73 | 0.767 | 1.71 |
| 3 | **RandomForest** | Tree Ensemble | 19.93 | 2.89 | 13.41 | 8.83 | 0.711 | 0.73 |
| 4 | **GaussianProcess** | Kernel | 20.55 | 2.45 | 15.79 | 12.44 | 0.697 | 0.22 |
| 5 | XGBoost | Boosting | 21.38 | 4.01 | 12.76 | 6.01 | 0.662 | 0.10 |
| 6 | LightGBM | Boosting | 21.85 | 2.76 | 16.19 | 11.48 | 0.656 | 0.09 |
| 7 | HistGradientBoosting | Tree Ensemble | 22.13 | 2.76 | 16.22 | 11.19 | 0.647 | 0.29 |
| 8 | KNN | Instance-based | 24.72 | 1.79 | 19.35 | 18.38 | 0.564 | 0.01 |
| 9 | Ridge | Linear | 29.57 | 2.27 | 24.91 | 23.03 | 0.376 | 0.19 |
| 10 | LinearRegression | Linear | 29.59 | 2.45 | 25.02 | 23.19 | 0.374 | 0.18 |
| 11 | Lasso | Linear | 30.19 | 2.28 | 25.22 | 23.09 | 0.351 | 0.19 |
| 12 | ElasticNet | Linear | 30.41 | 1.96 | 26.17 | 24.91 | 0.348 | 0.18 |
| 13 | SVR (RBF) | Kernel | 40.78 | 5.15 | 30.23 | 16.74 | **−0.166** | 0.02 |

**SVR is worse than a constant-mean baseline** (negative R²) at default hyperparameters —
almost certainly a default-hyperparameter artifact (`C=1.0`, `gamma='scale'` are a
well-documented SVR pitfall), not evidence the kernel/algorithm is fundamentally unsuited;
noted for completeness, not chased further per the "no extensive tuning" rule.

**Bold** models make up the top statistical tier — see
[`model_comparison_report.md`](model_comparison_report.md) for the Friedman/Nemenyi evidence
that separates "genuinely better" from "looks better but isn't distinguishable from noise."

## 4. Optional models: not included, with reasons

| Model | Status | Reason |
|---|---|---|
| **NGBoost** | Failed, not retried | Confirmed library incompatibility: `ngboost==0.5.11`'s `NGBRegressor.fit()` does not satisfy `scikit-learn==1.9.0`'s `check_is_fitted` protocol inside a `Pipeline` (verified directly: the model fits and predicts correctly in isolation, but `Pipeline.predict()` raises `NotFittedError` after a successful `Pipeline.fit()`). A version-compatibility issue between two independently-versioned libraries, not a bug in this project's code. |
| **EBM** (Explainable Boosting Machine) | Never completed | Killed twice by the environment's background-execution limits before finishing even one 50-fold benchmark pass (consistent with EBM's known slower training cost from automatic pairwise-interaction detection). Treated as computationally infeasible within this phase's budget, not as evidence of poor performance. |

Both are explicitly marked "optional... if computationally feasible" in the phase brief;
neither blocks the exit criteria.

## 5. Reproducing a single model

```python
from models import build_model_pipeline, run_repeated_cv, summarize
result = run_repeated_cv("ExtraTrees", X, y)
print(summarize(result))
```

## Next

See [`model_comparison_report.md`](model_comparison_report.md) for statistical
significance testing, model diversity, and feature-importance agreement across families;
[`residual_analysis_report.md`](residual_analysis_report.md) for residual diagnostics and
physical-plausibility checks; [`learning_curve_report.md`](learning_curve_report.md) for
bias/variance/data-sufficiency; and the Final Recommendation (which ≤4 model families
advance to Phase 5) at the end of `model_comparison_report.md`.
