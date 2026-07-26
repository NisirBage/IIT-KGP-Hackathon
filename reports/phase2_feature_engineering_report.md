# Phase 2 — Physics-Informed Feature Engineering & Validation Report

Companion documents: [`feature_registry.md`](feature_registry.md) (per-feature status),
[`redundancy_report.md`](redundancy_report.md), [`stability_report.md`](stability_report.md),
[`feature_benchmark_report.md`](feature_benchmark_report.md). Raw numbers:
[`phase2_analysis_results.json`](phase2_analysis_results.json). Code:
[`src/features/`](../src/features/) (reusable module), [`src/run_phase2_analysis.py`](../src/run_phase2_analysis.py)
(analysis harness — not part of the modeling pipeline itself).

24 candidate engineered features were implemented across 6 families (residence, temperature,
Arrhenius-inspired, flow, geometry, interactions). Every claim below traces to a specific
statistic in the JSON dump — nothing is asserted because it "sounds useful."

---

## 1. Feature engineering module

`src/features/` — one file per physical family (`residence.py`, `temperature.py`,
`arrhenius.py`, `flow.py`, `geometry.py`, `interactions.py`), each exporting pure,
deterministic `f(df) -> Series` functions with no fitted parameters and no target leakage.
`build.py` assembles the full candidate set and records each feature's raw-feature
"parents" (`PARENT_OF`), used throughout for incremental-value testing. `norm_residence` is
the one feature whose current implementation fits mean/std from whatever DataFrame is
passed in — flagged in its docstring as needing a proper train-fit-only transform in Phase 3
(the preprocessing pipeline phase), not a leak in the analysis done here (train-only).

## 2. Feature validation: correlation battery (all 24 candidates)

Full table: [`phase2_analysis_results.json → feature_validation`](phase2_analysis_results.json).
Headline groupings:

- **Strongest, most consistent group**: `avg_temp` / `arrhenius_avg` (identical Spearman
  −0.711, MI 0.417/0.418, dcor 0.670/0.670 — see §4 for why these two are near-duplicates),
  `max_temp_approx` (Spearman −0.694), `min_temp_approx` (Spearman −0.488) — every method
  agrees on direction and rough magnitude.
- **The non-monotonic signature reappears exactly as in Phase 1**: the entire
  `residence_proxy` family (5 features, all monotonic transforms of each other) and
  `avgtemp_x_residence` show near-null Pearson/Spearman/Kendall (all |r|<0.14, mostly
  non-significant) but **among the highest Mutual Information values in the whole table**
  (`residence_proxy` MI=0.339 [Phase 1] / 0.339 [Phase 2 recompute — consistent], `inv_residence`
  MI=0.311, `avgtemp_x_residence` MI=0.347). A feature with ~zero monotonic correlation but
  high MI is the textbook fingerprint of an inverted-U relationship.
- **`concentration_mol_L`-derived interactions stay null**: `residence_x_conc` shows the
  weakest correlations of any interaction term (Pearson 0.050 ns, MI 0.099) — consistent
  with Phase 1's finding that concentration carries no signal, even in combination.

## 3. Incremental value: nested F-tests + nonlinear (kNN) R² gain

Full table: [`phase2_analysis_results.json → incremental_value`](phase2_analysis_results.json).
This is the step that actually separates "correlated" from "adds new information beyond its
parents" — several important, sometimes counter-intuitive results:

| Feature | Partial F-test vs. parents | Verdict |
|---|---|---|
| `avg_temp`, `delta_T` (vs. raw `[inlet_T, jacket_T]`) | R² identical (0.4055→0.4055), F≈0, p=1.0 | **Zero linear incremental value** — exact linear combinations. Expected; their value is elsewhere (see below). |
| `max_temp_approx` | R² 0.4055→0.4187, p=0.070 (borderline) | Weak linear signal, but **largest kNN CV R² gain of any feature (+0.033)** — a nonlinear (max) function a linear model can't exploit even when hand-fed the feature, but a nonparametric model can. |
| `abs_delta_T` (vs. `delta_T` alone) | R² 0.017→0.044, **p=0.042 (significant)** | Confirms the Phase 1 "V-shaped" finding (zero-yield rows exist at both extreme +ΔT and extreme −ΔT) with a proper nested test, not just the 6-row spot-check from Phase 1. |
| `norm_delta_T` (vs. `[delta_T, avg_temp]`) | R² 0.408→0.439, **p=0.0037** | The *relative* gradient (ΔT normalized by mean temperature) carries real information beyond the raw gradient and mean level combined. |
| **`arrhenius_inlet`** (vs. `inlet_temperature_K` alone) | R² 0.164→0.261, **p=2.1e-5** | **Strongest incremental result in the whole table.** Important methodological point (§4): Spearman/Kendall cannot detect this at all, because they are invariant to monotonic transforms — only a nested R² test that cares about functional *shape*, not rank order, can see it. |
| `arrhenius_avg` (vs. `avg_temp` alone) | R² 0.405→0.416, p=0.109 (ns) | Unlike `arrhenius_inlet`, applying the same transform to the already-strong `avg_temp` adds little — there's less room to improve on an R²=0.405 baseline. |
| `severity_index` (vs. `[residence_proxy, delta_T]`) | R² 0.022→0.047, **p=0.050** (borderline; p=0.0067 when `avg_temp` is also in the parent set, per Phase 1 §5) | Sensitive to which parents are controlled for, but consistently on the "real, not spurious" side across two independent test specifications. |
| `residence_proxy` (vs. raw `[length_m, flow_rate_L_min]`) | R² 0.008→0.019, p=0.201 (ns) | **The ratio entered linearly does not help a linear model** — consistent with its own near-zero linear R² (0.006, §5 shape diagnostics). Its value requires either a quadratic term or a nonlinear model, not a plain linear term. |
| `residence_x_conc`, `L2_over_F`, `L_times_deltaT`, `residence_x_temp` | All p>0.34 | No incremental value beyond their parents — **rejected**. |

## 4. A key methodological finding: rank correlation can't judge transform value

`arrhenius_inlet` is a strictly monotonic transform of `inlet_temperature_K`
(`exp(-1000/T)` is monotonically increasing in T), so by construction its Spearman/Kendall
correlation with the target is **identical** to raw `inlet_temperature_K`'s (both exactly
−0.378 / −0.238). Phase 1 used this similarity to tentatively flag the Arrhenius transform
as "likely redundant." **The nested F-test proves that assessment wrong**: the transform
adds highly significant incremental linear R² (p=2.1e-5) and has the single largest
quadratic-shape gain of any feature tested (§5, +0.089). The lesson, applied going forward:
*monotonic-invariant correlation measures (Spearman, Kendall) can never detect whether a
nonlinear transform of a variable is worth including over its raw form* — only a
model-based nested comparison can. Feature registry statuses have been corrected
accordingly (`arrhenius_inlet` promoted from "Pending" to "Validated").

## 5. Nonlinearity / shape diagnostics (linear R² vs. quadratic R²)

Full table: [`phase2_analysis_results.json → shape_diagnostics`](phase2_analysis_results.json).
Figure: [`figures/phase2/nonlinearity_lowess_top8.png`](figures/phase2/nonlinearity_lowess_top8.png)
(LOWESS fits over the 8 highest-distance-correlation features).

| Shape classification | Features | Evidence |
|---|---|---|
| Near-zero *quadratic* gain (misleading if taken as "linear" — see correction below) | `avg_temp` (+0.002), `arrhenius_avg` (+0.001), `max_temp_approx` (+0.001), `abs_delta_T` (+0.0000006) | quadratic R² ≈ linear R² |
| **Meaningfully nonlinear / non-monotonic** | `arrhenius_inlet` (**+0.089**, largest gain of all), `inv_flow` (+0.062), `log_flow` (+0.054), `temp_ratio` (+0.023), `delta_T` (+0.023), `min_temp_approx` (+0.020), `residence_proxy` and its transform-siblings (+0.017) | quadratic term roughly triples or more the explained variance vs. linear alone |

**Correction after visually inspecting the LOWESS fits (`figures/phase2/nonlinearity_lowess_top8.png`)
— do not trust the quadratic-gain number alone for shape classification.** A symmetric
parabola is a poor basis for a one-sided saturating curve, so "near-zero quadratic gain" can
mean either "genuinely linear" *or* "nonlinear in a shape a parabola can't capture" — the
number alone can't distinguish these. Looking at the actual LOWESS curves: `avg_temp`,
`arrhenius_avg`, and `max_temp_approx` all show a **clear saturating/threshold (sigmoidal)
shape**, not a straight line — yield holds around 65–70% for `avg_temp` below ~410 K, then
drops sharply through roughly 410–480 K, then plateaus near zero above that. This is a much
more physically specific and informative finding than "linear decline": it is the continuous
counterpart of the zero-yield point-mass finding from Phase 1 — **the reactor doesn't
degrade gradually with temperature, it collapses over a fairly narrow critical band**. This
directly supports evaluating a threshold/hurdle-aware model in Phase 4 (§9), and means a
plain linear term for `avg_temp`, while statistically the single strongest predictor
available, is still leaving a known, visually obvious nonlinearity on the table — a spline,
a logistic-shaped basis, or a tree-based model's native ability to place a split near the
~410–480 K band is expected to outperform a linear term here. `min_temp_approx` and
`arrhenius_inlet`'s LOWESS curves are noisier and closer to a broad, weaker version of the
same declining-then-flattening pattern; `severity_index` and `severity_index_arrhenius`
(lower dcor, bottom row of the figure) show visibly noisier LOWESS wiggle that should not be
over-interpreted as real sub-structure given how scattered the underlying point cloud is.

This separates the "needs a threshold/spline-aware representation" features (the whole
average-temperature cluster) from the "needs a nonlinear representation for a different
reason — non-monotonicity" features (residence family) — both actionable for Phase 3/4:
tree-based models should be able to exploit the temperature threshold natively, while linear
models will need either a spline/piecewise term for `avg_temp` or acceptance of a real,
known gap versus tree ensembles on this dataset.

## 6. Sequential feature selection (forward, Ridge, 8 features, CV RMSE)

Selected, in order: `inv_residence → min_temp_approx → temp_ratio → delta_T →
norm_delta_T → arrhenius_inlet → arrhenius_avg → flow_x_deltaT`.

**Notably, `avg_temp` and `max_temp_approx` were never selected** despite being the
strongest standalone correlates. This is not a contradiction — it's forward selection
correctly detecting that once `min_temp_approx` and `delta_T` are already in the model,
`avg_temp` is *exactly* linearly reconstructable from them (`avg = min + ΔT/2`) and adds
zero marginal R². **This is the clearest possible illustration of why "highest individual
correlation" and "belongs in the final multivariate feature set" are different questions** —
reinforces why the redundancy report's cluster-representative approach (§ below) is the
right way to build the final set, not a simple top-N-by-correlation ranking.

## 7. Redundancy analysis

See [`redundancy_report.md`](redundancy_report.md) in full. Summary: 24 candidates collapse
into **6 clusters** under hierarchical clustering (|Spearman| ≥ 0.7 merge threshold), several
with VIF = ∞ in the full 24-feature design (exact linear reconstructability). One
representative per cluster, plus 2 singleton features with independently validated
incremental value (`abs_delta_T`, `arrhenius_inlet`), survive redundancy screening.

## 8. Stability analysis

See [`stability_report.md`](stability_report.md) in full. 500-bootstrap resampling shows the
temperature-level cluster (`avg_temp`/`arrhenius_avg`/`max_temp_approx`) is essentially
noise-free (100% top-8 appearance, ~6% coefficient of variation on the correlation
estimate). `residence_proxy`'s near-0% top-8 appearance by Spearman is explained, not
contradicted, by its MI-based stability (~21% CV, in line with other features) — a rank
metric structurally cannot stabilize around a non-monotonic relationship.

## 9. Zero-yield mechanism, revisited with engineered features only

Using **only** 6 engineered physics features (`avg_temp, residence_proxy, delta_T,
severity_index, max_temp_approx, abs_delta_T`) — no raw columns — four simple classifiers
all clear the 75.3% majority-class baseline by 9–14 points with AUC ≥ 0.886 (full table in
[`feature_benchmark_report.md`](feature_benchmark_report.md)). The depth-3 decision tree's
actual learned rule set is a remarkably clean, physically-legible summary of the entire
Phase 0–2 investigation:

```
avg_temp <= 449.01                                  (cooler regime)
├── severity_index <= 77.64                          → non-zero yield
└── severity_index > 77.64
    ├── abs_delta_T <= 71.92                          → non-zero yield
    └── abs_delta_T > 71.92                           → ZERO YIELD (extreme severity + extreme gradient, even while cooler)
avg_temp > 449.01                                    (hotter regime)
├── residence_proxy <= 0.17                           → non-zero yield  (short residence protects even when hot)
└── residence_proxy > 0.17
    ├── delta_T <= 24.67                               → non-zero yield
    └── delta_T > 24.67                                → ZERO YIELD  (hot + long residence + still actively heating)
```

This is strong evidence — **not proof of mechanism, but a strong empirical signal** — that
the reactor genuinely operates in two structurally distinct regimes separable by simple
physics features alone. **Per the phase objective, this is flagged as justification to
evaluate a two-stage (classify-then-regress / hurdle) architecture in Phase 4**, not a
decision to build one now.

## 10. Lightweight benchmarking (Ridge / RandomForest / CatBoost, no tuning)

Full results and discussion: [`feature_benchmark_report.md`](feature_benchmark_report.md).
Headline: the curated 5-feature validated set (`avg_temp, residence_proxy, residence_sq,
delta_T, severity_index`) is best-or-tied-best for **all three model families** against both
raw-only and the full 24-candidate dump; dumping in everything never wins and measurably
hurts Ridge. A follow-up test adding the two later-validated features
(`abs_delta_T`, `arrhenius_inlet`) showed no decisive change (all deltas within CV noise).

---

## 11. Rejected features

| Feature | Why it failed | Statistical evidence | Redundancy-driven? | Reconsider? |
|---|---|---|---|---|
| `norm_residence` | Pure rescaling of `residence_proxy` (z-score) | Spearman/Kendall/MI **identical** to `residence_proxy`; pairwise r=1.0 | Yes — exact duplicate by construction | No — never useful unless a specific model requires standardized inputs, and that's a preprocessing step, not a new feature |
| `residence_sq`, `log_residence`, `inv_residence` alone | Same rank-correlation profile as `residence_proxy` (monotonic transforms); no significant incremental F-test individually (`log_residence` p=0.94, `inv_residence` p=0.50) | See §3 table | Yes — cluster with `residence_proxy` | `residence_sq` reconsidered as a *paired* term with `residence_proxy` (not standalone) for linear models needing the quadratic shape (§5) |
| `residence_x_temp`, `residence_x_conc`, `L2_over_F`, `L_times_deltaT` | No significant incremental value beyond stated parents (all p>0.34) | §3 table | Partially — high pairwise correlation with `residence_proxy` cluster | No — clearly dominated by simpler alternatives already in the set |
| `flow_sq` | Weak standalone correlation (dcor 0.182, barely above raw flow's 0.189) and non-significant incremental F-test (p=0.192) | §2, §3 | No | No |
| `temp_ratio` | Near-perfectly anti-correlated with `norm_delta_T` (ρ=−1.0) and with `delta_T` (ρ=−0.998) — pure restatement | Redundancy report | Yes | No — `delta_T`/`norm_delta_T` are simpler and equally informative |
| `avgtemp_x_residence`, `residence_x_temp`, `L2_over_F`, `severity_index_arrhenius` | Numerically dominated by `residence_proxy` in this dataset's design (pairwise ρ 0.93–0.995 with it) rather than encoding genuine multiplicative interaction | Redundancy report; contrast with `severity_index` which *did* pass a nested test despite similar-looking pairwise correlation to `delta_T` | Yes | `severity_index_arrhenius` specifically: reconsider only if a future version uses fitted (not fixed-constant) Arrhenius parameters, since its current weak showing may be an artifact of the arbitrary `ARRHENIUS_C=1000` scale choice, not the concept itself |
| `arrhenius_avg` | Statistically real (Spearman −0.711) but adds no incremental value beyond `avg_temp` alone (p=0.109) and near-zero quadratic gain (+0.0007) — functionally redundant with `avg_temp` | §3, §5 | Yes — same cluster as `avg_temp` | No — `avg_temp` is simpler and equally predictive |
| Flow-transform family (`inv_flow`, `flow_sq`, `log_flow`) as a group | None shows a standalone correlation meaningfully above raw `flow_rate_L_min` itself | §2 | Partially (internally correlated with each other) | Low priority — flow rate appears to matter to yield mainly through the `residence_proxy` ratio, not on its own |

---

## 12. Final ranked feature table

| Rank | Feature | Evidence Score* | Physical Plausibility | Incremental Value | Recommendation |
|---|---|---|---|---|---|
| 1 | `avg_temp` | Very high (5/5 correlation methods significant; 100% bootstrap stability) | High — effective reaction temperature | None beyond raw temps for *linear* models; essential for tree models (§10 benchmark) | **Use** |
| 2 | `residence_proxy` | High (near-zero monotonic corr, but MI in top-3 of all 24 candidates and stable) | High — direct τ proxy, matches series-reaction theory | Weak as a plain linear term; needs `residence_sq` or a nonlinear model to realize | **Use** (paired with `residence_sq` for linear models) |
| 3 | `residence_sq` | Moderate (F-test p=0.111 vs. `residence_proxy` alone — borderline) | High — required to represent the τ_opt interior maximum | Marginal on its own, but necessary complement to rank 2 for linear models | **Use** (as a pair with `residence_proxy`, not standalone) |
| 4 | `delta_T` | Moderate (Spearman −0.24, p=0.003; huge effect size on zero-split, Phase 1) | High — net heating boundary condition | None vs. raw temps for linear models; large benefit for zero-split classification | **Use** |
| 5 | `severity_index` | Moderate (2 independent nested F-tests, p=0.0067 and p=0.050) | High — Damköhler-type severity proxy | Confirmed incremental beyond `avg_temp`/`delta_T`/`residence_proxy` | **Use** |
| 6 | `arrhenius_inlet` | High incremental (p=2.1e-5, largest quadratic gain +0.089) | Medium — literal Arrhenius form but arbitrary constant | Strongest incremental result in Phase 2 | **Use if the model family benefits** (benchmark showed no decisive CV gain when added — see §10); include for the pitch's scientific narrative even if the RMSE delta is noise-level |
| 7 | `abs_delta_T` | Moderate (F-test p=0.042 vs. `delta_T` alone) | Medium — captures the "extreme in either direction" V-shape found in Phase 1 | Confirmed but small in absolute R² terms | Optional — same "no decisive CV gain" caveat as rank 6 |
| — | Everything else (17 features) | Low/none | Varies | None or redundant | **Rejected** — see §11 |

\*"Evidence Score" is qualitative here (high/moderate/low), built from the combination of
correlation-battery significance, incremental F-test results, and bootstrap stability —
deliberately not collapsed into a single numeric score, since the report's own findings
(§4, §6) show that different single metrics disagree about the same feature for
structurally understandable reasons; a single number would hide exactly the nuance this
report exists to surface.

---

## Final Decision

**Recommendation: Option 3 — a reduced engineered feature subset.**

Concretely: **raw 5 features + `avg_temp`, `residence_proxy`, `residence_sq`, `delta_T`,
`severity_index`** (10 total) as the primary modeling feature set for Phase 3 onward.
`arrhenius_inlet` and `abs_delta_T` are retained in the registry as **validated-but-optional**
— defensible by their own statistics, but not shown to move CV RMSE beyond noise once
combined with the core 5, and the phase's explicit mandate is to prefer the smaller set
absent a decisive reason not to.

This is **not** "raw + validated" as a blanket rule (Option 2), because the validation
process itself rejected 19 of 24 candidates — the surviving 5 (or optionally 7) are already
the reduced/curated outcome of that process, not merely "everything that passed one test."
It is also **not** "raw only" (Option 1): the benchmark (§10) shows raw+validated beats
raw-only for every model tested, most dramatically for CatBoost (19.75→17.76 RMSE) and
RandomForest (21.61→19.91).

We are **not** recommending Option 4 (separate feature sets per model family) at this
stage: while Ridge shows near-zero benefit from the engineered features (because `avg_temp`
and `delta_T` are exact linear combinations of the raw temperatures), it is also not *hurt*
by including them, and maintaining one shared feature set is simpler to defend and to
reproduce. This should be revisited if Phase 4's fuller model comparison shows a starker
linear-vs-nonlinear divergence.

**Supporting evidence, in one place:**
1. Benchmark: raw+validated beats raw-only and raw+all-24 for every model tested (§10).
2. Redundancy: 24 candidates cluster into 6 groups; the recommended 5 are exactly one
   representative per cluster plus the two features (`residence_sq`, `severity_index`) with
   independently confirmed incremental value (§7).
3. Stability: none of the recommended 5 shows fragile, resample-sensitive correlation (§8).
4. Zero-yield separability: the recommended set's core members (`avg_temp,
   residence_proxy, delta_T, severity_index`) achieve 87–89% accuracy / AUC 0.89–0.94
   classifying the zero-yield regime alone — strong evidence the set captures the dominant
   structural signal in this dataset (§9).

## Next step

Phase 3: build the leakage-safe preprocessing pipeline (proper train-fit-only scaling for
`norm_residence`-style transforms, if retained; target-transform evaluation given the
25% zero-inflation; final feature-set serialization) using exactly this 10-feature
recommended set as the default, with the 2 optional features available behind a config flag
for ablation in Phase 4.
