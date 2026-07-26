# Phase 4 — Model Comparison Report

RMSE ranking alone is not treated as sufficient anywhere in this report — every claim of
"better" is backed by a paired test (all 13 models share the identical 50-fold sequence,
seed=42, so paired comparisons are valid) or is explicitly labeled as *not* statistically
distinguishable.

## 1. Omnibus test: are the 13 models different at all?

**Friedman test** (non-parametric, rank-based, appropriate for >2 related samples across
the same 50 folds): **χ² = 530.2, p = 8.2×10⁻¹⁰⁶**. Overwhelming evidence that at least
some models differ — this licenses the pairwise post-hoc analysis below (a Friedman test
this significant is expected whenever a linear-model tier and a tree-model tier coexist in
the same comparison; the real question is which of the *close* pairs are distinguishable).

## 2. Pairwise vs. the best model (ExtraTrees) — paired t-test, Wilcoxon, bootstrap CI

| Model | Mean RMSE diff vs. ExtraTrees | Paired t p | Wilcoxon p | Bootstrap 95% CI | 95% CI excludes 0? |
|---|---|---|---|---|---|
| CatBoost | +1.29 | 1.5e-8 | 1.4e-8 | [0.93, 1.67] | Yes |
| RandomForest | +3.23 | 2.1e-17 | 2.5e-14 | [2.75, 3.72] | Yes |
| GaussianProcess | +3.86 | 3.0e-16 | 7.6e-14 | [3.24, 4.51] | Yes |
| XGBoost | +4.69 | 1.0e-13 | 1.9e-12 | [3.80, 5.55] | Yes |
| LightGBM | +5.16 | 1.6e-21 | 1.8e-15 | [4.54, 5.76] | Yes |
| HistGradientBoosting | +5.43 | 4.0e-23 | 1.8e-15 | [4.86, 6.02] | Yes |
| KNN | +8.03 | 5.5e-32 | 1.8e-15 | [7.48, 8.58] | Yes |
| LinearRegression | +12.90 | 3.8e-34 | 1.8e-15 | [12.14, 13.72] | Yes |
| Ridge | +12.87 | 3.0e-36 | 1.8e-15 | [12.18, 13.61] | Yes |
| Lasso | +13.50 | 3.6e-37 | 1.8e-15 | [12.80, 14.22] | Yes |
| ElasticNet | +13.72 | 9.2e-39 | 1.8e-15 | [13.06, 14.40] | Yes |
| SVR (RBF) | +24.09 | 9.0e-34 | 1.8e-15 | [22.56, 25.61] | Yes |

**Every single model, including CatBoost, is statistically significantly worse than
ExtraTrees in a direct paired comparison** — even the +1.29 RMSE gap to CatBoost has a 95%
CI that excludes zero ([0.93, 1.67]). Taken alone, this would suggest ExtraTrees is
unambiguously the single best model. **§3 (Nemenyi) complicates this conclusion in an
important way — read on before concluding "ExtraTrees, full stop."**

## 3. Nemenyi post-hoc (multiple-comparison-corrected) — the more conservative, more honest test

The pairwise tests above each ask "is A different from B" in isolation; running 12+
pairwise tests without correction inflates the false-positive rate. Nemenyi corrects for
testing all 78 pairs among the 13 models simultaneously — and several "significant" pairwise
results above **do not survive that correction**:

| Comparison | Nemenyi p | Distinguishable? |
|---|---|---|
| ExtraTrees vs. CatBoost | 0.953 | **No** — despite the significant paired t-test above |
| ExtraTrees vs. RandomForest | 0.018 | Yes |
| ExtraTrees vs. GaussianProcess | 0.0038 | Yes |
| CatBoost vs. RandomForest | 0.660 | **No** |
| CatBoost vs. GaussianProcess | 0.363 | **No** |
| RandomForest vs. GaussianProcess | 1.000 | **No** |
| CatBoost vs. XGBoost | 0.033 | Yes (borderline) |
| RandomForest vs. XGBoost | 0.983 | **No** |
| GaussianProcess vs. XGBoost | 0.999 | **No** |
| XGBoost vs. LightGBM vs. HistGradientBoosting (all pairs) | 0.94 – 1.00 | **No** — mutually indistinguishable |
| KNN vs. HistGradientBoosting | 0.660 | **No** (surprising given the 2.6-point raw RMSE gap) |
| Linear family (all 4, all pairs) | 0.97 – 0.9999 | **No** — mutually indistinguishable, regularization choice doesn't matter here |
| SVR vs. everything except Lasso/ElasticNet | ~0.0 | Yes — SVR is reliably the worst |

**Reading this correctly**: there is a genuine top cluster — **{ExtraTrees, CatBoost,
RandomForest, GaussianProcess}** — where ExtraTrees is significantly ahead of RandomForest
and GaussianProcess specifically, but **not** significantly ahead of CatBoost, and CatBoost
itself is not significantly ahead of RandomForest or GaussianProcess either. There is a
second, mutually-indistinguishable cluster — **{XGBoost, LightGBM, HistGradientBoosting}** —
each significantly behind the top cluster's leaders but not cleanly separable from each
other or (surprisingly) always from KNN. The full pairwise matrix:
[`phase4_nemenyi_pvalues.csv`](phase4_nemenyi_pvalues.csv).

## 4. Model diversity (for future ensemble design, not acted on this phase)

Figure: [`figures/phase4/model_diversity.png`](figures/phase4/model_diversity.png) —
prediction correlation and error (residual) correlation across all 13 models.

- **The entire tree/boosting cluster is highly inter-correlated** (0.94–0.997 prediction
  correlation; e.g. HistGradientBoosting↔LightGBM = 0.997, ExtraTrees↔CatBoost = 0.99,
  RandomForest↔XGBoost = 0.98). Per the phase's own framing: **these models are unlikely to
  improve each other through stacking** — they are largely learning the same function.
- **The linear family correlates highly with itself** (Ridge↔LinearRegression = 0.99,
  ElasticNet↔Lasso = 0.997) but comparatively weakly with the tree cluster — the single
  lowest pairwise prediction correlation in the whole matrix is
  **ElasticNet↔GaussianProcess (0.696)**, closely followed by **Lasso↔GaussianProcess
  (0.696)** and **XGBoost↔SVR (0.694)**.
- **Practical implication for Phase 6/7**: if ensembling is pursued, a tree/boosting model
  paired with a linear model (or GaussianProcess) offers real diversity; pairing two members
  of the tree/boosting cluster with each other does not, regardless of how good either looks
  individually on the leaderboard.

## 5. Feature importance agreement across model families

Full numbers: [`phase4_importance.json`](phase4_importance.json). Figure:
[`figures/phase4/feature_importance_comparison.png`](figures/phase4/feature_importance_comparison.png).
**Correction applied mid-analysis**: permutation importance must be evaluated on held-out
data — ExtraTrees/CatBoost/RandomForest all reach near-zero in-sample RMSE (confirmed:
ExtraTrees in-sample RMSE = 2.3×10⁻¹³, i.e. the model has memorized the training set), so
computing permutation importance on the same data used for fitting gives a baseline score of
~0 and wildly inflated, meaningless "importance" values (tens of RMSE units). All permutation
and SHAP numbers below are computed on an 80/20 held-out split instead.

**Raw-feature permutation importance (RMSE degradation when shuffled, held-out data)** — the
ranking is **identical across all four models tested** (ExtraTrees, CatBoost, RandomForest,
Ridge): `jacket_temperature_K` > `inlet_temperature_K` > `flow_rate_L_min` ≈ `length_m` >
`concentration_mol_L` (indistinguishable from zero — even slightly negative for Ridge,
the signature of a genuinely uninformative feature). **This is strong, convergent,
model-independent confirmation of the Phase 1/2 EDA conclusions** — now validated by actual
trained-model sensitivity, not just correlation.

**Impurity importance (tree/boosting models, engineered 10-feature set)**: all three
agree `avg_temp` dominates (29–51% of total importance), followed by `inlet_temperature_K`,
with `concentration_mol_L` consistently smallest.

**SHAP — where a real, worth-investigating disagreement appears**: for the three tree/
boosting models, `avg_temp` is overwhelmingly the top SHAP feature (13.3–20.2 mean |SHAP|).
**For Ridge, `avg_temp`'s SHAP value is only 1.68 — smallest of the engineered features —
while `residence_sq` (28.1) and `residence_proxy` (20.5) are the *largest* SHAP values of
any feature for any model in this table**, despite Phase 2 finding `residence_proxy` has
essentially *zero* linear/monotonic correlation with the raw target (Pearson 0.076, ns).

**Investigating the disagreement**: this is very likely a multicollinearity/coefficient-
instability artifact, not a genuine physical signal Ridge uniquely detected. Two mechanisms
plausibly combine: (1) `avg_temp` is an exact linear combination of the two raw
temperatures (Phase 3 §5), so once Ridge fits coefficients on `inlet_temperature_K` and
`jacket_temperature_K` directly, `avg_temp`'s own coefficient carries little *additional*
weight — its explanatory power is already absorbed elsewhere, which SHAP correctly reflects
as a small marginal contribution; (2) `residence_proxy` and `residence_sq` are, by
construction, highly correlated with each other (one is the square of the other) — Ridge's
L2 penalty does not perform feature selection the way Lasso does, so it can assign large,
partially-offsetting coefficients to a correlated pair without being penalized much for
their redundancy, inflating each one's individual SHAP attribution even though their
*combined* contribution to any given prediction may be modest. **This is flagged as a
genuine limitation of interpreting Ridge's individual engineered-feature coefficients on
this feature set**, not a discovery that residence terms secretly matter more than
temperature for a linear model — the tree-model consensus (`avg_temp` dominant,
`concentration_mol_L` negligible) is trusted as the more reliable read on physical drivers,
consistent with every correlation-based analysis since Phase 1.

## 6. Final Recommendation — promote exactly 4 model families to Phase 5

| Model | Why chosen | Optimization ceiling | Hyperparameters worth tuning | Anticipated risks |
|---|---|---|---|---|
| **ExtraTrees** | Best RMSE (16.69), not significantly beaten by anything (§2/§3), **zero physically-implausible predictions** (residual_analysis_report.md §2), best fold-to-fold stability of the top tier | Learning curve shows validation RMSE still falling at n=120 with near-zero training error (residual_analysis / learning_curve reports) — high variance, meaningful room via depth/leaf regularization *and* more data | `max_depth`, `min_samples_leaf`, `max_features`, `n_estimators` | Already near-perfectly memorizes training folds (in-sample RMSE ≈0) — Phase 5 tuning must actively fight overfitting, not chase more training-set fit |
| **CatBoost** | Statistically indistinguishable from ExtraTrees (Nemenyi p=0.953) — genuinely competitive, not just "close on paper" | Gradient boosting typically has a larger default-to-tuned gap than bagging; learning curve shows the same still-falling, near-zero-train-error pattern as ExtraTrees | `learning_rate`, `depth`, `l2_leaf_reg`, `iterations` (with early stopping) | 7.3% physically-implausible predictions (min prediction −5.0) — needs post-hoc clipping to [0,100] regardless of tuning outcome; slowest model benchmarked (1.7s/fit vs. <0.5s for the bagging models) — budget Phase 5 search time accordingly |
| **RandomForest** | Statistically tied with CatBoost and GaussianProcess (not distinguishable from either), **also zero physically-implausible predictions**, structurally different bias/variance profile than ExtraTrees (learning curve: non-zero, higher training RMSE ≈8 vs. ExtraTrees' ≈0 — meaningfully less overfit already at defaults) | Smaller gap between current defaults and its likely ceiling than ExtraTrees (already less overfit), but validation RMSE is also still falling at n=120 | `max_depth`, `min_samples_leaf`, `max_features` | Highly correlated with ExtraTrees' predictions (0.97) — limited incremental value if both are tuned and later ensembled together; tune primarily as a plausibility-safe alternative to ExtraTrees, not expecting it to separately beat CatBoost |
| **GaussianProcess** | Statistically tied with CatBoost/RandomForest tier; **lowest prediction/error correlation with the entire tree cluster among competitive models** (§4) — the one candidate that offers real ensemble diversity if Phase 6/7 pursues stacking | Kernel choice (currently `RBF + WhiteKernel`, both untuned) has a large, well-documented effect on GP performance; meaningful tuning ceiling | Kernel length-scale / variance bounds, `alpha` (noise), kernel family (Matérn vs. RBF) | **16% physically-implausible predictions — worst of any competitive model** (range −18.1 to +112.6); noisy, high-variance training-fit behavior at small sample sizes (learning curve, std=6.8 at n=24). Both risks are real but standard/fixable (bounded-output post-processing, more principled kernel/prior choice) — not disqualifying, but must be addressed explicitly in Phase 5, not carried forward silently |

### Not promoted, with reasons

| Model | Status | Reason |
|---|---|---|
| XGBoost | **Keep for reference** | Not significantly worse than RandomForest or GaussianProcess by Nemenyi, but shows the worst fold-to-fold instability of any top-8 model (highest `rmse_std`=4.01, highest max per-sample std=24.07 across repeats) — the phase's own principle ("reject unstable models even if RMSE is slightly better") argues against a tuning slot when CatBoost already represents boosting in the top 4 and is both better-performing and more stable at defaults. Revisit only if CatBoost tuning underperforms expectations in Phase 5. |
| LightGBM, HistGradientBoosting | **Keep for reference** | Statistically indistinguishable from XGBoost and from each other (all pairwise Nemenyi p≥0.94); both clearly behind the top cluster. Plausible explanation: at n=150, boosting's typical advantage over bagging hasn't had a chance to manifest at default (untuned) learning rates/iteration counts — genuinely possible either would close the gap with tuning, but the ≤4 cap and CatBoost's already-dominant position in the boosting family make that a lower-priority bet than the 4 selected. |
| KNN | **Keep for reference** | Clearly and significantly behind the top tier (both paired and Nemenyi tests vs. ExtraTrees/CatBoost); limited hyperparameter surface (`k`, distance metric, weighting) offers a low optimization ceiling. Inherently bounded predictions (0% implausible) and simplicity make it a reasonable interpretability baseline, not a tuning candidate. |
| LinearRegression, Ridge, Lasso, ElasticNet | **Keep for reference / possible ensemble component** | Mutually indistinguishable from each other (regularization choice doesn't matter — the ceiling is the model class, not under-tuning) and far behind every non-linear model. Lowest prediction correlation with the tree cluster of any reasonably-behaved model family (§4) — worth keeping *un-tuned* as a diversity component if stacking is attempted in Phase 6/7, but not worth a Phase 5 tuning slot on its own. |
| SVR (RBF) | **Reject** | R²=−0.166 at defaults — worse than predicting the mean. Statistically the worst model tested against nearly everything. Almost certainly a default-hyperparameter artifact (`C`, `gamma` are notoriously impactful for RBF-SVR) rather than proof the algorithm is unsuitable, but with 4 already-strong, already-validated candidates occupying every promotion slot, revisiting SVR is not worth the effort. |
| NGBoost | **Reject (infeasible)** | Confirmed library version incompatibility (`baseline_model_report.md` §4) — not a performance judgment. |
| EBM | **Reject (infeasible)** | Never completed a benchmark pass within the environment's runtime budget — not a performance judgment. |

## Exit check

Every one of the 13 successfully-benchmarked models was evaluated under identical CV,
identical preprocessing-per-family, and identical metrics; the shortlist above is justified
by paired statistical tests *and* the more conservative Nemenyi correction *and* physical
plausibility *and* stability — not by raw RMSE rank alone. Phase 5 can begin hyperparameter
optimization on the 4 selected families without revisiting this baseline comparison.
