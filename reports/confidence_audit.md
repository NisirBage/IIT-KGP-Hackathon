# Confidence Audit

Every major conclusion from this project, graded honestly. **Proven** = directly, repeatedly
demonstrated with no significant caveat. **Strongly supported** = clear statistical evidence,
one or more named limitations. **Plausible** = consistent with evidence, not independently
verified. **Speculative** = offered as a reasonable hypothesis, evidence is weak or absent.

| # | Claim | Evidence | Confidence | Supporting Experiments |
|---|---|---|---|---|
| 1 | The zero-yield mass (25% of training rows) is a genuine structural regime, not a rounding artifact | Hard gap between exact 0 and smallest non-zero (0.013); GMM BIC drop 1→2 components (1529→1135); later confirmed classifier-separable (87-93% accuracy) using only physics features | **Strongly supported** | EXP-000, Phase 1 §1, Phase 2 §9 |
| 2 | `concentration_mol_L` has no effect on yield | 5/5 independent correlation methods agree (Pearson/Spearman/Kendall/MI/dcor all ≈0); confirmed dead-last by permutation importance across 4 model families | **Proven** (for "no detectable effect in this data") | EXP-000, EXP-012 |
| 3 | `avg_temp` is the dominant physical driver of yield | Strongest of 24 candidates by every correlation method; dominant impurity/SHAP feature across ExtraTrees/CatBoost/RandomForest; consistent raw-feature permutation ranking (jacket_temp>inlet_temp) across 4 model families | **Proven** | EXP-000, EXP-012 |
| 4 | Residence time has a non-monotonic relationship with yield (matches τ_opt theory) | Near-zero linear/monotonic correlation but 2nd-highest MI of any candidate — the statistical signature of non-monotonicity; not directly fitted/confirmed via an explicit τ_opt curve | **Plausible** | EXP-000, Phase 2 §2 |
| 5 | `avg_temp`'s relationship with yield is sigmoidal (threshold), not linear | LOWESS visualization shows a clear flat→steep→flat pattern; not formalized with a statistical changepoint test | **Plausible** | Phase 2 §5 |
| 6 | Preprocessing must be fit inside the CV loop, not on the full dataset first | Directly demonstrated: leaky evaluation is optimistically biased in 29/30 independent reseeded trials | **Proven** | Phase 3 leakage validation |
| 7 | Tree-based models (RandomForest, CatBoost, ExtraTrees) are scale-invariant | RMSE identical to 3 decimal places across every scaler tested | **Proven** (mechanically expected, empirically confirmed) | EXP-003 |
| 8 | KNN performs best with no scaling on this dataset | One scaler benchmark, not independently replicated; KNN not used in the final model | **Plausible** (unreplicated, low-stakes) | EXP-003 |
| 9 | A top statistical tier of 4 models (ExtraTrees, CatBoost, RandomForest, GaussianProcess) outperforms the remaining 9 benchmarked models | Friedman χ²=530.2, p=8.2e-106; Nemenyi post-hoc separates this tier from the rest consistently | **Strongly supported** | EXP-008, EXP-009 |
| 10 | ExtraTrees is unambiguously the single best individual model | Best raw RMSE, but Nemenyi shows it's **not** statistically distinguishable from CatBoost (p=0.953) despite a significant raw paired test (p=1.5e-8) | **Not supported as stated** — see #9 for the defensible version | EXP-009 |
| 11 | Bagged-tree models (ExtraTrees, RandomForest) structurally respect the [0,100] physical bound; other model families regularly violate it | 0% implausible predictions for both, at every tested configuration, vs. 7-18% for boosting/kernel/linear models | **Proven** | EXP-010 |
| 12 | Hyperparameter tuning improved CatBoost, RandomForest, and GaussianProcess | Paired t-test + Wilcoxon on identical final-protocol folds, p≤0.003 for all three | **Proven** | EXP-015 |
| 13 | Hyperparameter tuning made ExtraTrees worse, not better | Same protocol, p=0.046 (t-test) / p=0.025 (Wilcoxon), 95% CI excludes zero in the "worse" direction | **Proven** | EXP-015 |
| 14 | RepeatedKFold(5,10) is an appropriate, non-optimistically-biased validation protocol for this dataset | Reseed-stability std=0.24 across 20 reseeds; matches honest nested-CV estimate within 0.03 RMSE (smaller than either protocol's own std) | **Strongly supported** | EXP-013 |
| 15 | The naive (non-nested) hyperparameter-selection estimate is pessimistically, not optimistically, biased at n=150 | Directly measured (naive=19.201 > honest=16.664); mechanism (smaller inner-fold training sets) identified and consistent with independently-observed learning curves | **Strongly supported** (for this dataset; contradicts textbook default expectation) | EXP-013 |
| 16 | The 3-model linear blend (ExtraTrees+CatBoost+RandomForest) genuinely outperforms ExtraTrees alone, in-distribution | 10-way independent robustness check (same qualitative pattern in all 10); leave-one-repeat-out nested CV (14.974±0.622, consistent with every other estimate); paired test vs. ExtraTrees p<0.000001, 95% CI=[−2.03,−1.85] | **Proven** (in-distribution, on this dataset) | EXP-018, EXP-019, EXP-020 |
| 17 | The ensemble's negative RandomForest coefficient reflects a genuine "shrinkage-correction" mechanism, not overfitting noise | Consistent sign and magnitude across 10 independent training-repeat choices; large, stable RMSE gain on every held-out check | **Strongly supported** (mechanism explanation itself is a **plausible** hypothesis, not independently proven) | EXP-019 |
| 18 | The ensemble will generalize safely to data meaningfully different from the training distribution | No test performed on any data outside the original 150-row training population | **Speculative / untested** | None |
| 19 | Clipping predictions to [0,100] improves rather than merely constrains accuracy | Direct comparison: clipped LOO RMSE (14.762) beats unclipped (14.974) | **Proven** | EXP-019 |
| 20 | The frozen inference pipeline is deterministic and leakage-free | 3/3 identical SHA-256 hashes across independent runs; adversarial testing of 4 malformed-input and 2 malformed-output cases (one real bug found and fixed) | **Proven** | Phase 7 reproducibility audit |
| 21 | The pipeline will run correctly on an independent judge's machine | Exact package versions pinned (`requirements_frozen.txt`); never tested in a second, independently-created environment | **Plausible** (strong indirect evidence, no direct confirmation) | None |
| 22 | Feature selection (which of 24 candidates to keep) did not leak target information into the final CV performance estimates | Not tested — features were selected using statistics computed on the full training set, not nested inside nested CV | **Known limitation, not defended** | None (explicitly named gap) |
| 23 | The physical mechanism (over-reaction via k2-dominance) explains the zero-yield regime | Consistent with all observed patterns (thermal/residence clustering); alternative explanations (solver artifact) not ruled out; no independent kinetic verification | **Plausible** | Phase 1 §6, Phase 2 §9 |
| 24 | Test-set rows predicted as exactly zero represent genuine recognition of the training-identified collapse regime, not arbitrary model behavior | Predicted-zero test rows have mean avg_temp=467.6K vs. training zero-yield-group mean=469.1K (1.5K gap) — a striking match; delta_T pattern is less consistent, noted as a caveat | **Strongly supported** (cannot be proven without true test labels) | Phase 8 ad hoc analysis (see `presentation_outline.md` backup slide) |

## Reading this table

Rows 1, 2, 3, 6, 7, 9 (as reframed), 11, 12, 13, 16, 19, 20 are the load-bearing claims of
the final submission — all **Proven** or **Strongly supported**, each with a specific,
reproducible experiment behind it. Rows 18, 21, 22 are the honest, named limitations that
should be volunteered proactively if not asked, not surfaced only under pressure (see
`technical_defense.md` Part A for the full reasoning behind each).
