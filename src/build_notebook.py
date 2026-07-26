"""Builds the presentation-ready competition notebook (Phase 7 deliverable) as valid
nbformat v4 JSON, without depending on the `nbformat` package. Narrative content
summarizes each phase's headline findings (citing the full reports, not duplicating them);
the final 2 sections are genuinely executable and run the real, frozen inference pipeline.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "competition_notebook.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


cells = []

cells.append(md(
    "# Reactor Yield Prediction — Competition Notebook",
    "",
    "**Team submission for the ML Hackathon: The Predictive Modeling Optimization Challenge**",
    "",
    "This notebook is the presentation-ready narrative summary of the full project. Every claim",
    "below is backed by a detailed report in `reports/` and reproducible code in `src/` — this",
    "notebook cites and summarizes that work rather than re-deriving it, except for the final",
    "inference section, which genuinely executes the frozen submission pipeline live.",
    "",
    "**Final model**: a 3-model linear blend (ExtraTrees + CatBoost + RandomForest), clipped to",
    "[0,100]. **Validated RMSE**: 14.76 ± 0.64 (leave-one-repeat-out nested CV), a statistically",
    "decisive improvement (p<0.000001) over the best single model (ExtraTrees, RMSE 16.69).",
))

# 1. Problem overview
cells.append(md(
    "## 1. Problem Overview",
    "",
    "The reactor is a non-isothermal, continuous-flow system running a series reaction network:",
    "",
    "```",
    "A --k1--> B (desired)  --k2--> C (side product)",
    "```",
    "",
    "Both `k1` and `k2` are Arrhenius rate constants (temperature-dependent). The task: predict",
    "`overall_yield` (% of product B) from 5 raw operating conditions",
    "(`flow_rate_L_min, concentration_mol_L, inlet_temperature_K, length_m, jacket_temperature_K`),",
    "replacing an expensive CFD/BVP simulator with a fast ML surrogate.",
    "",
    "Key physics-derived hypotheses established before touching the data (full reasoning:",
    "[`reports/phase0_problem_understanding.md`](../reports/phase0_problem_understanding.md)):",
    "- Yield vs. residence time should be **non-monotonic** (an interior maximum — too short and A",
    "  hasn't converted; too long and B has already degraded to C).",
    "- Temperature has a **dual effect**: raises both the desired and side reaction rates.",
    "- `jacket_temperature_K − inlet_temperature_K` (net thermal driving force) was hypothesized as",
    "  more informative than either raw temperature alone.",
))

# 2. Dataset audit
cells.append(md(
    "## 2. Dataset Audit",
    "",
    "Full details: [`reports/dataset_passport.md`](../reports/dataset_passport.md),",
    "[`reports/phase1_eda_findings.md`](../reports/phase1_eda_findings.md).",
    "",
    "- **150 training rows, 50 test rows**, 5 raw features, zero missing values, zero duplicate",
    "  rows, zero impossible physical values.",
    "- **25% of training targets are exactly zero** (37/150) — verified as a genuine structural",
    "  regime, not a rounding artifact (a hard gap exists: nothing between 0 and 0.013, then a",
    "  normal continuous tail). GMM modality analysis confirms a two-regime mixture.",
    "- **Train/test distributions largely overlap** (KS test finds no significant shift on any",
    "  feature), though PSI flags `length_m` specifically as a moderate-shift concern — this",
    "  informed favoring repeated CV over a single holdout throughout the project.",
    "- **Zero-yield rows cluster at high jacket temperature and long residence time** — the first",
    "  quantitative signal of the thermal-collapse mechanism explored further in Phase 2.",
))

# 3. Physics understanding / feature engineering
cells.append(md(
    "## 3-4. Physics Understanding & Feature Engineering",
    "",
    "Full details: [`reports/phase2_feature_engineering_report.md`](../reports/phase2_feature_engineering_report.md),",
    "[`reports/feature_registry.md`](../reports/feature_registry.md).",
    "",
    "24 candidate engineered features were built and evaluated across correlation battery,",
    "incremental-value (nested F-test), redundancy (VIF + hierarchical clustering), bootstrap",
    "stability, and LOWESS-based shape diagnostics. **5 features were validated and promoted**:",
    "",
    "| Feature | Formula | Why it matters |",
    "|---|---|---|",
    "| `avg_temp` | `(inlet_T + jacket_T)/2` | Strongest single predictor by every method; LOWESS reveals a **sigmoidal collapse** (not linear) around 410-480K |",
    "| `residence_proxy` | `length_m / flow_rate_L_min` | Near-zero linear correlation but 2nd-highest Mutual Information — the textbook signature of the hypothesized non-monotonic τ_opt relationship |",
    "| `residence_sq` | `residence_proxy²` | Required to represent the interior-maximum shape in a linear model |",
    "| `delta_T` | `jacket_T − inlet_T` | Net thermal driving force; large effect specifically on the zero-yield split |",
    "| `severity_index` | `residence_proxy × delta_T` | Damköhler-type interaction term; passed two independent nested-F tests despite 0.90 pairwise correlation with `delta_T` alone |",
    "",
    "**A key methodological finding**: Spearman/Kendall correlation cannot detect whether a",
    "*nonlinear transform* of a variable adds value over its raw form (both have identical rank",
    "correlation by construction) — only a nested model comparison can. This is why",
    "`arrhenius_inlet` (`exp(-1000/T)`) looked \"redundant\" by correlation but passed the",
    "strongest incremental F-test in the whole feature set (p=2.1e-5).",
))

# 5. Preprocessing
cells.append(md(
    "## 5. Preprocessing",
    "",
    "Full details: [`reports/preprocessing_report.md`](../reports/preprocessing_report.md),",
    "[`reports/leakage_validation_report.md`](../reports/leakage_validation_report.md).",
    "",
    "Every preprocessing step (feature construction, scaling, transforms) lives inside a",
    "single `sklearn.Pipeline`, refit independently on every CV fold — never fit once on the",
    "full dataset. This was empirically validated, not just asserted: a leaky-vs-correct",
    "comparison across 30 independent reshufflings showed the leaky approach is optimistically",
    "biased in **29/30 cases** (mean bias −0.29 RMSE).",
    "",
    "**Scaling is genuinely model-specific, not universal** — measured, not assumed:",
    "",
    "| Model | Scaler | Why |",
    "|---|---|---|",
    "| Ridge / GaussianProcess | `power_yeojohnson` | Both meaningfully benefit from distribution normalization |",
    "| KNN | **none** | Counter-intuitively, every scaler tested made KNN *worse* — the raw, unscaled feature magnitudes appear informative for this dataset's distance structure |",
    "| RandomForest / CatBoost / ExtraTrees | none | Confirmed scale-invariant to 3 decimal places across every scaler tested |",
))

# 6. Model benchmarking
cells.append(md(
    "## 6. Model Benchmarking",
    "",
    "Full details: [`reports/baseline_model_report.md`](../reports/baseline_model_report.md),",
    "[`reports/model_comparison_report.md`](../reports/model_comparison_report.md).",
    "",
    "13 model families benchmarked under an identical `RepeatedKFold(5,10)` protocol (50 folds,",
    "same seed, same folds for every model — enabling valid paired statistical tests):",
    "",
    "| Rank | Model | RMSE | R² |",
    "|---|---|---|---|",
    "| 1 | ExtraTrees | 16.69 ± 2.24 | 0.798 |",
    "| 2 | CatBoost | 17.99 ± 2.48 | 0.767 |",
    "| 3 | RandomForest | 19.93 ± 2.89 | 0.711 |",
    "| 4 | GaussianProcess | 20.55 ± 2.45 | 0.697 |",
    "| ... | (9 more families) | | |",
    "| 13 | SVR (RBF) | 40.78 ± 5.15 | −0.166 |",
    "",
    "**Friedman test**: χ²=530.2, p=8.2e-106 — real differences exist. **Nemenyi post-hoc**",
    "(multiple-comparison corrected) overturns a naive pairwise conclusion: ExtraTrees vs.",
    "CatBoost looks significant in isolation (p=1.5e-8) but is **not** distinguishable after",
    "correcting for testing all 78 pairs (p=0.953) — a top statistical tier of 4 models",
    "{ExtraTrees, CatBoost, RandomForest, GaussianProcess} was promoted, not just the top-1.",
    "",
    "**Physical plausibility mattered as much as RMSE**: only the two bagged-tree models",
    "(ExtraTrees, RandomForest) produce zero predictions outside [0,100] — a structural property",
    "of bagging, not tuning. GaussianProcess (16% implausible) and most boosting/linear models",
    "regularly violate the physical bound at default settings.",
))

# 7. Hyperparameter optimization
cells.append(md(
    "## 7. Hyperparameter Optimization",
    "",
    "Full details: [`reports/validation_strategy_report.md`](../reports/validation_strategy_report.md),",
    "[`reports/hyperparameter_optimization_report.md`](../reports/hyperparameter_optimization_report.md),",
    "[`reports/final_model_selection_report.md`](../reports/final_model_selection_report.md).",
    "",
    "Optuna (TPE sampler, SQLite-checkpointed, MedianPruner) tuned all 4 shortlisted families.",
    "**Every tuned configuration was re-validated under the full `RepeatedKFold(5,10)` protocol**",
    "(not the lighter search-time CV) before any \"tuned beats baseline\" claim was accepted —",
    "this caught something important:",
    "",
    "| Model | Baseline RMSE | Tuned RMSE | Significant? |",
    "|---|---|---|---|",
    "| **ExtraTrees** | 16.693 | **16.871 (worse)** | **Yes — tuning hurt it** (p=0.046) |",
    "| CatBoost | 17.987 | 17.207 (better) | Yes (p<0.0001) |",
    "| RandomForest | 19.926 | 19.324 (better) | Yes (p=0.002) |",
    "| GaussianProcess | 20.550 | 19.734 (better) | Yes (p<0.0001) |",
    "",
    "**ExtraTrees — the best baseline model — got statistically significantly worse after",
    "tuning.** The lighter per-trial CV budget used during search was noisy enough to select a",
    "configuration that didn't generalize. **Final selection: ExtraTrees at Phase 4 defaults**,",
    "not the Optuna-suggested configuration — evidence overruled the tuning process's own output.",
))

# 8. Ensemble evaluation
cells.append(md(
    "## 8. Ensemble Evaluation",
    "",
    "Full details: [`reports/ensemble_evaluation_report.md`](../reports/ensemble_evaluation_report.md),",
    "[`reports/final_submission_recommendation.md`](../reports/final_submission_recommendation.md).",
    "",
    "Prediction correlation among the tuned models was very high (0.92-0.99), suggesting limited",
    "ensemble potential. **Disagreement-rate analysis told a different story**: >10-point",
    "disagreement is only 2.7-9.3% for ExtraTrees/CatBoost/RandomForest pairs, but 31-36% for any",
    "pair involving GaussianProcess — concentrated in the 410-480K thermal-transition region.",
    "",
    "A linear blend of all 4 models produced a striking ~1.7 RMSE-point improvement. The fitted",
    "coefficients included a **negative weight on RandomForest** — exactly the kind of",
    "multicollinearity instability flagged for Ridge in Phase 4, so this was **not accepted at",
    "face value**. It was stress-tested: refit independently on each of 10 repeats (the same",
    "qualitative pattern appeared in all 10), then validated with a fully rigorous",
    "leave-one-repeat-out nested CV. **The result held**: RMSE 14.97 (unclipped) → **14.76 after",
    "clipping to [0,100]**, which also fixed a real cost (24.7% implausible predictions",
    "unclipped → 0% after clipping, with *improved* RMSE, not a trade-off).",
    "",
    "GaussianProcess's contribution proved statistically negligible (p=0.615 to drop it) — the",
    "**final ensemble uses only 3 models**: ExtraTrees + CatBoost + RandomForest.",
    "",
    "**Final comparison**: blend RMSE 14.76 ± 0.64 vs. ExtraTrees alone 16.69 ± 2.24",
    "— diff=−1.93, paired t-test p<0.000001, 95% CI=[−2.03,−1.85]. Gains are broad-based across",
    "zero-yield, thermal-transition, and high-yield regions alike, not concentrated in one.",
))

# 9 & 10: final inference + submission generation (EXECUTABLE)
cells.append(md(
    "## 9. Final Inference",
    "",
    "This section genuinely executes the frozen submission pipeline — nothing above this point",
    "is re-run (it would take hours; see `reports/experiment_registry.md` for the full compute",
    "log). Everything below is live.",
))
cells.append(code(
    "import sys\n",
    "from pathlib import Path\n",
    "sys.path.insert(0, str(Path('..') / 'src'))\n",
    "\n",
    "from inference.pipeline import run_inference\n",
    "\n",
    "report = run_inference()\n",
    "print('All validations passed:', report['all_passed'])\n",
    "print('Output SHA-256:', report['output_sha256'])\n",
    "print('Prediction summary:', report['prediction_validation']['summary_stats'])"
))
cells.append(md(
    "## 10. Submission Generation",
    "",
    "The cell above already wrote `submission/TeamName.csv`. Confirming its contents meet the",
    "exact competition specification:",
))
cells.append(code(
    "import pandas as pd\n",
    "\n",
    "sub = pd.read_csv('../submission/TeamName.csv')\n",
    "print('Shape:', sub.shape, '(expected: (50, 1))')\n",
    "print('Columns:', list(sub.columns), \"(expected: ['overall_yield'])\")\n",
    "print('Dtype:', sub['overall_yield'].dtype)\n",
    "print('Range:', sub['overall_yield'].min(), '-', sub['overall_yield'].max())\n",
    "sub.head(10)"
))
cells.append(md(
    "---",
    "",
    "**End of notebook.** Full validation, reproducibility, and risk-assessment detail:",
    "[`reports/submission_validation_report.md`](../reports/submission_validation_report.md),",
    "[`reports/reproducibility_report.md`](../reports/reproducibility_report.md),",
    "[`reports/competition_readiness_report.md`](../reports/competition_readiness_report.md).",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {NB_PATH} ({len(cells)} cells)")
