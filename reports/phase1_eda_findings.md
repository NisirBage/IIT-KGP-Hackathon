# Phase 1A — Evidence-Driven EDA Findings

Every number in this document comes from [`src/run_phase1_eda.py`](../src/run_phase1_eda.py)
(full dump: [`phase1_eda_results.json`](phase1_eda_results.json)) plus two targeted follow-up
checks run directly in the venv (redundancy correlations and a nested-model F-test — commands
preserved in the session log, results reproduced verbatim below). Figures are in
[`reports/figures/`](figures/). Per the ground rule for this phase: **every claim below is
labeled with the statistic that supports it; nothing is asserted from visual impression alone.**

---

## 1. Zero-yield verification

**Question:** is the 24.67% (37/150) mass at `overall_yield = 0` a real phenomenon, a rounding
artifact, or a distribution modeling illusion?

| Check | Result |
|---|---|
| Exact zeros | 37 / 150 (24.67%) |
| Values in (0, 0.001) | 0 |
| Values in (0, 0.01) | 0 |
| Values in (0, 0.1) | 3 |
| Values in (0, 1.0) | 20 |
| Smallest non-zero value | **0.013** |

**Finding: this is a genuine point mass, not a rounding artifact.** If zero were an artifact of
rounding very small simulator outputs, we'd expect a smooth accumulation of values just above
zero (0.0001, 0.0003, 0.0009, …) trailing into the exact-zero bin. Instead there is a **hard
gap**: nothing at all between 0 and 0.013, then a normal-looking continuous tail from 0.013
upward. That gap is the signature of a genuine structural discontinuity in the underlying
process (e.g. the BVP solver returning/clipping to a true zero once B is fully consumed) rather
than float rounding.

**Modality check (Gaussian-mixture BIC scan)** — lower BIC is better:

| k components | BIC on full target | BIC on non-zero subset only (n=113) |
|---|---|---|
| 1 | 1529.3 | 1146.8 |
| 2 | 1135.0 | 1102.4 |
| 3 | 1108.1 | 1014.4 |

On the **full target**, going from 1→2 components buys a huge BIC improvement (Δ394), but
2→3 buys almost nothing further (Δ27). That is the expected signature of a **two-regime
mixture** — a "collapsed" mass near zero plus a separate continuous "active" regime — not a
smoothly unimodal distribution.

On the **non-zero subset alone**, the 2→3 improvement (Δ88) is actually larger than the 1→2
improvement (Δ45), which is a weaker, more ambiguous signal — GMM component counts on 1-D
data almost always "improve" with more components, so this does **not** by itself prove a
second sub-mode inside the active regime. Read it as: *no strong standalone evidence of extra
structure within the non-zero yields*, beyond what's explained by the features (§4 correlation
analysis already explains a good chunk of that spread).

**Conclusion (evidence-graded):** the zero mass is a real, structurally distinct regime
(**high confidence** — clean gap + large 1→2 BIC drop). Whether it represents true "reactor
extinction" (complete over-reaction, B → C) versus some other simulator branch (e.g. a
solver-stability cutoff) is a physical interpretation, not yet a proven mechanism — see the
Challenge Table (§6) for the competing explanations and the evidence for each.

**Decisive follow-up check — gradient direction vs. absolute temperature.** Of the 37
zero-yield rows, 6 actually have *negative* `delta_T_jacket_inlet` (net **cooling**, jacket
colder than inlet) — directly at odds with a pure "jacket over-heats the reactor" story. All
6 of those rows nonetheless have high `avg_temp` (mean 465.9 K) driven by an already-hot
**inlet** temperature (mean 483.3 K) and above-average `residence_proxy` (0.734 vs. the
dataset mean of 0.567) — i.e., the feed enters hot and stays long enough to over-react even
while the jacket is (mildly) cooling it. This is **direct, decisive evidence favoring
"absolute thermal level" over "net heating gradient direction"** as the operative mechanism,
and it upgrades the corresponding Challenge Table row (§6) from Medium to
**Medium-High** confidence.

Figure: [`figures/target_distribution.png`](figures/target_distribution.png) — full
histogram, a zoom into [0,5), and a log-scale y-axis view. The log-scale panel makes the
"cliff" between the zero bar and the next-smallest values visually obvious.

---

## 2. Train vs. test distribution comparison

Mandatory before any feature engineering: if test comes from a different operating regime,
CV performance on train will not transfer.

| Feature | KS stat | KS p | Wasserstein dist. | PSI |
|---|---|---|---|---|
| flow_rate_L_min | 0.140 | 0.436 | 3.35 | 0.230 |
| concentration_mol_L | 0.140 | 0.436 | 0.17 | 0.224 |
| inlet_temperature_K | 0.127 | 0.565 | 5.85 | 0.236 |
| length_m | 0.087 | 0.931 | 0.66 | 0.337 |
| jacket_temperature_K | 0.153 | 0.326 | 10.99 | 0.165 |

**Two tests disagree, and that disagreement is itself the finding.** The KS test says "no
significant difference" for every feature (all p > 0.32) — but KS has weak power at n=150
vs n=50, so a null result here is weak evidence of similarity, not strong evidence of it. PSI
(quantile-bin based, standard thresholds: <0.10 no shift, 0.10–0.25 moderate, >0.25
"significant") tells a moderately more cautious story: **`length_m` (PSI=0.337) crosses into
the "significant shift" zone**, and `flow_rate_L_min`, `concentration_mol_L`, and
`inlet_temperature_K` all sit in the upper end of "moderate" (0.22–0.24). Only
`jacket_temperature_K` (0.165) is comfortably "moderate-low."

**Practical read:** there is no red-alert evidence of train/test coming from disjoint
operating regimes (ranges overlap, KS is null, Wasserstein distances are small relative to
each feature's own spread), but PSI's consistent "moderate" reading across 4/5 features — with
`length_m` outright crossing the significance line — is enough to **not fully trust a single
train/validation split**. This directly shapes Phase 5:

- **Validation strategy:** favor repeated / nested CV over one static holdout, so the reported
  RMSE reflects variability across many resamplings rather than luck of one split matching
  (or not matching) the test distribution on `length_m`.
- **Feature engineering:** prefer ratio/difference features (`residence_proxy`,
  `delta_T_jacket_inlet`) that are somewhat scale-invariant over raw `length_m`, since the
  raw feature shows the largest shift signal.
- **Model selection:** favor models with some built-in regularization/robustness (tree
  ensembles with shrinkage, ElasticNet) over high-variance/no-regularization options, since a
  model that overfits exact `length_m` boundaries in train has the most exposure here.
- **Confidence in CV scores:** treat the local CV RMSE as a plausible-range estimate, not a
  guaranteed number — build in a safety margin when comparing candidate models that are close
  in CV score.

Figure: [`figures/train_test_comparison.png`](figures/train_test_comparison.png) —
histogram+KDE, boxplot, and ECDF per feature, train vs test overlaid.

---

## 3. Statistical validation: zero-yield vs. non-zero-yield groups

Descriptive means are not enough on their own — every comparison below is backed by a
parametric test (Welch's t, robust to unequal variance), a non-parametric test (Mann–Whitney
U, robust to non-normality), two effect sizes (Cohen's d for magnitude, Cliff's delta for
non-parametric/rank effect size), and a bootstrap 95% CI on the mean difference (5,000
resamples).

| Feature | Mean (zero grp) | Mean (nonzero grp) | Welch p | Mann–Whitney p | Cohen's d | Cliff's δ | 95% CI (mean diff) |
|---|---|---|---|---|---|---|---|
| flow_rate_L_min | 30.12 | 43.85 | **0.0006** | **0.0010** | −0.64 (medium) | −0.36 (medium) | [−21.0, −6.0] |
| concentration_mol_L | 2.15 | 2.36 | 0.237 | 0.221 | −0.21 (small) | −0.13 (small) | [−0.56, 0.16] (crosses 0) |
| inlet_temperature_K | 439.0 | 419.6 | **0.023** | **0.028** | 0.43 (medium) | 0.24 (small) | [3.2, 35.3] |
| length_m | 15.81 | 13.52 | 0.046 | 0.113 | 0.33 (small) | 0.17 (small) | [0.06, 4.46] |
| **jacket_temperature_K** | **499.2** | **423.6** | **1.4e-16** | **1.9e-12** | **1.70 (huge)** | **0.77 (large)** | [61.3, 89.4] |
| residence_proxy | 0.758 | 0.505 | 0.013 | **7.8e-5** | 0.41 (medium) | 0.43 (medium) | [0.05, 0.44] |
| delta_T_jacket_inlet | 60.22 | 3.93 | **4.9e-6** | **1.8e-5** | 0.85 (large) | 0.47 (medium) | [34.0, 78.2] |
| **avg_temp** | **469.1** | **421.6** | **5.6e-13** | **1.1e-11** | **1.61 (huge)** | **0.75 (large)** | [37.1, 57.5] |
| severity_index | 49.45 | −4.51 | 5.2e-6 | 1.2e-7 | 1.26 (large) | 0.58 (large) | [34.7, 74.7] |
| inv_inlet_temp | 0.00230 | 0.00241 | 0.025 | 0.028 | −0.43 (medium) | −0.24 (small) | crosses 0 in raw units, mirrors inlet_temperature_K |

**Reading the discrepancies (this is where rigor matters most):**

- **`concentration_mol_L` shows no significant difference by any test** (Welch p=0.24, MWU
  p=0.22, |d|=0.21). This directly confirms — with actual statistics, not just chemical
  intuition — the Phase 0 hypothesis that inlet concentration does not independently drive
  the yield-collapse regime.
- **`length_m` is the one genuinely ambiguous result**: Welch's t says borderline-significant
  (p=0.046) but Mann–Whitney does not (p=0.113), and both effect sizes are small (d=0.33,
  δ=0.17). A parametric/non-parametric disagreement like this, on a small-effect result, is a
  sign the "significance" is fragile (likely driven by a few points, or by
  `length_m`'s asymmetric role — it only matters *combined* with flow rate, not alone). **We
  should not treat `length_m` alone as a validated driver of the zero-yield split.**
  `residence_proxy` — which combines it with flow rate — tells a much cleaner story (MWU
  p=7.8e-5, a full 3 orders of magnitude more significant than `length_m` alone), consistent
  with the theoretical prediction that residence time, not reactor length by itself, is the
  physically meaningful quantity.
- **`jacket_temperature_K` alone has the single largest effect size of anything tested**
  (Cohen's d=1.70), larger even than the purpose-built `delta_T_jacket_inlet` (d=0.85). This
  is an important, slightly uncomfortable finding for the "net heating gradient" story from
  Phase 0 — see the Challenge Table (§6) for why, and the redundancy analysis in §5 for the
  quantitative explanation (jacket temperature and the ΔT feature are themselves correlated
  at ρ=0.77 in this dataset, because inlet and jacket temperatures were sampled nearly
  independently).

Figure: [`figures/zero_vs_nonzero_violins.png`](figures/zero_vs_nonzero_violins.png) — violin
plots of all 10 candidate quantities split by zero/non-zero group.

---

## 4. Correlation analysis vs. target (why rankings differ by method)

| Feature | Pearson r | Spearman ρ | Kendall τ | Mutual Info | Distance Corr |
|---|---|---|---|---|---|
| flow_rate_L_min | 0.038 (ns) | 0.111 (ns) | 0.085 (ns) | 0.094 | 0.189 |
| concentration_mol_L | 0.009 (ns) | 0.040 (ns) | 0.028 (ns) | 0.000 | 0.068 |
| inlet_temperature_K | −0.405\*\*\* | −0.378\*\*\* | −0.238\*\*\* | 0.192 | 0.446 |
| length_m | 0.080 (ns) | 0.013 (ns) | 0.013 (ns) | 0.096 | 0.134 |
| jacket_temperature_K | −0.498\*\*\* | −0.595\*\*\* | −0.418\*\*\* | 0.189 | 0.502 |
| residence_proxy | 0.076 (ns) | **−0.079 (ns)** | −0.032 (ns) | **0.339** | 0.212 |
| delta_T_jacket_inlet | −0.129 (ns) | −0.240\*\* | −0.161\*\* | 0.105 | 0.194 |
| **avg_temp** | **−0.637\*\*\*** | **−0.711\*\*\*** | **−0.517\*\*\*** | **0.417** | **0.670** |
| severity_index | −0.193\* | −0.280\*\*\* | −0.192\*\*\* | 0.122 | 0.240 |
| inv_inlet_temp | 0.375\*\*\* | 0.378\*\*\* | 0.238\*\*\* | 0.201 | 0.425 |

(\* p<0.05, \*\* p<0.01, \*\*\* p<0.001; ns = not significant)

**Why the rankings disagree between methods — and why that disagreement is informative:**

- **Pearson/Spearman/Kendall only detect linear or monotonic relationships.** Mutual
  information and distance correlation can detect *any* statistical dependence, including
  non-monotonic ones. When a feature scores near-zero on the first three but meaningfully
  higher on MI/dcor, that's a direct fingerprint of a **non-monotonic relationship** — exactly
  what the series-reaction theory (Phase 0, §2) predicts for residence time.
- **`residence_proxy` is the clearest case of this signature in the whole dataset**: Pearson
  0.076, Spearman **−0.079**, Kendall −0.032 — all statistically null, and the sign even
  flips between Pearson and Spearman. Yet its Mutual Information (0.339) is the
  **second-highest of every feature tested**, only behind `avg_temp`. A feature with zero
  monotonic correlation but high MI is, almost definitionally, evidence of an **inverted-U /
  interior-maximum relationship** — this is a genuine quantitative confirmation of the
  τ_opt theory from Phase 0, not an assumption.
- **`avg_temp` is the strongest predictor by every single method** (Pearson −0.637, Spearman
  −0.711, Kendall −0.517, MI 0.417, dcor 0.670) — and Spearman being notably stronger than
  Pearson in magnitude (−0.711 vs −0.637) says the relationship, while monotonic (declining),
  is not purely linear.
- **`concentration_mol_L` is null on every single method** (MI is exactly 0.000, dcor a low
  0.068) — the strongest and most consistent "no effect" signal in the table, reinforcing the
  §3 finding with an independent line of evidence.
- MI estimates use k-NN (Kraskov) entropy estimation, which is noisy at n=150 — treat MI as a
  **directional** ranking signal, not a precise effect-size number.

Figure: [`figures/correlation_heatmap.png`](figures/correlation_heatmap.png) (raw
features + target, Spearman), [`figures/target_vs_features.png`](figures/target_vs_features.png)
(scatter of yield against each raw feature, colored red for zero-yield rows — visually
confirms the zero-yield points cluster at high jacket_temperature_K and are scattered rather
than clustered on `concentration_mol_L`), [`figures/pairplot.png`](figures/pairplot.png).

---

## 5. Redundancy / collinearity of engineered thermal features

Two follow-up checks (not in the main script, run directly to resolve an open question from
§3/§4 about whether the engineered thermal features add real information or just repackage
`jacket_temperature_K`):

**(a) Spearman correlations among the thermal quantities:**

| | jacket_T | avg_temp | delta_T |
|---|---|---|---|
| jacket_T | 1.000 | 0.763 | 0.768 |
| avg_temp | 0.763 | 1.000 | 0.190 |
| delta_T | 0.768 | 0.190 | 1.000 |

`jacket_temperature_K` correlates ~0.77 with **both** `avg_temp` and `delta_T` individually —
this is expected and mechanical: `inlet_temperature_K` and `jacket_temperature_K` were sampled
almost independently in this dataset (raw feature-feature Spearman = 0.002, see
[`dataset_passport.md`](dataset_passport.md) collinearity note), so both linear combinations
inherit most of their variance from `jacket_temperature_K`. Importantly, **`avg_temp` and
`delta_T` are only weakly correlated with each other (ρ=0.190)** — they are not redundant
*with each other*, even though each is redundant-ish with the raw jacket temperature alone.

**(b) Exact-redundancy check for linear models** (does an engineered linear combination add
information beyond having both raw temperatures already in the model?):

| Model | R² |
|---|---|
| `[jacket_T, inlet_T]` | 0.4055 |
| `[jacket_T, inlet_T, delta_T]` | 0.4055 (identical) |
| `[jacket_T, inlet_T, avg_temp]` | 0.4055 (identical) |

Confirms the mathematical expectation exactly: for a **linear** model, `avg_temp` and
`delta_T` are exact linear combinations of the two raw temperatures and add *zero* new
information. Their value is specifically for (i) tree-based models, which cannot express a
difference/average across two features without wasting several splits, and (ii) models with
per-feature regularization/selection, where a single well-chosen combined axis is more
likely to be selected/kept than two individually-weaker raw features. This will be tested
directly in Phase 4 by comparing linear vs. tree-based model performance with/without these
engineered terms.

**(c) Is `severity_index` just repackaging `delta_T`?** Pairwise Spearman correlation between
`severity_index` and `delta_T` is **0.903** — very high, raising exactly this concern (since
`delta_T` has far more dynamic range than `residence_proxy`, the product `residence_proxy ×
delta_T` is numerically dominated by `delta_T`). We resolved this with a **nested-model
partial F-test** rather than trusting the pairwise correlation alone:

| Model | R² |
|---|---|
| `[avg_temp, delta_T, residence_proxy]` | 0.4082 |
| `[avg_temp, delta_T, residence_proxy, severity_index]` | 0.4376 |

Incremental F(1, 145) = 7.57, **p = 0.0067** — statistically significant improvement.
**Conclusion: despite the high pairwise correlation with `delta_T`, `severity_index` carries
real incremental information** (consistent with it representing a genuine multiplicative
interaction, i.e. a Damköhler-like severity term, rather than being a redundant restatement
of `delta_T`). This is the standard we're holding every registry entry to — pairwise
correlation flags a *concern*, only a nested/partial test resolves it.

---

## 6. Challenge every conclusion

| Observation | Leading explanation | Alternative explanation | Evidence | Confidence |
|---|---|---|---|---|
| 37/150 rows have exact zero yield | Genuine physical/simulator regime collapse (over-reaction, B→C complete) | Rounding/clipping artifact of the simulator's numerical output | Hard gap: 0 values in (0, 0.01), smallest non-zero = 0.013; GMM BIC drops sharply 1→2 components | **High** that it's a real distinct regime; **Medium** on the specific "complete over-reaction" mechanism (plausible, not directly provable from this data alone) |
| Zero-yield rows have much higher `jacket_temperature_K` (Cohen's d=1.70) | Absolute temperature level (`avg_temp`, d=1.61) drives over-reaction, regardless of heating/cooling direction | Net heating gradient (`delta_T_jacket_inlet`, d=0.85) drives over-reaction via accelerated k2 | 6/37 zero-yield rows have *negative* delta_T (net cooling) yet still show high `avg_temp` (465.9 K) via a hot inlet (483.3 K) + long residence — collapse happens without a positive gradient at all | **Medium-High** favoring absolute thermal level; gradient direction is not required for collapse, though most collapse cases (31/37) do co-occur with positive delta_T. Full disentangling still wants Phase 8 partial dependence/SHAP |
| `concentration_mol_L` shows no relationship with yield (all 5 correlation methods null, group-diff tests non-significant) | First-order series kinetics: yield *fraction* is independent of inlet concentration | Feature is measured/sampled too coarsely (121/150 unique values vs ~150 for others) to reveal a real but weak effect | 5-method correlation agreement (Pearson 0.009, Spearman 0.040, MI 0.000, dcor 0.068) + non-significant group tests (p=0.24, 0.22) | **High** for "no detectable linear/monotonic/general-dependence effect in this sample"; **Medium** on the specific first-order-kinetics mechanism |
| `residence_proxy` has ~null monotonic correlation but 2nd-highest MI (0.339) | Non-monotonic (inverted-U) yield-vs-residence-time relationship, matching series-reaction τ_opt theory | Two or more unrelated subgroups in the data each show a different (monotonic) sub-relationship, which MI picks up but isn't truly one nonlinear curve | Sign flip between Pearson (+0.076) and Spearman (−0.079); high MI without monotonic correlation is the textbook non-monotonic signature | **Medium-High** — consistent with theory; a definitive test needs a plotted/fitted curve shape (Phase 2) rather than correlation numbers alone |
| `length_m` alone: Welch p=0.046 but Mann-Whitney p=0.113 for zero-split | Not a real standalone effect — length only matters jointly with flow rate (as residence time) | Genuinely borderline effect that a larger sample would resolve either way | Parametric/non-parametric disagreement + small effect sizes (d=0.33, δ=0.17) vs. `residence_proxy`'s much stronger, more consistent signal (MWU p=7.8e-5) | **Medium-High** in favor of "not a real standalone effect" |
| PSI flags moderate-to-significant train/test shift on 4/5 features, but KS test finds none | Small-sample KS test has low power; PSI's quantile-binning is a more sensitive (if noisier) shift indicator here | PSI is unstable/over-sensitive with only 50 test points in ~10 bins (~5 points/bin) — the "shift" is bin-count noise, not real | Both explanations are statistically plausible with n=50; overlapping ranges + small Wasserstein distances argue against a severe shift | **Medium** — enough uncertainty to justify conservative validation choices (§2), not enough to declare a confirmed shift |
| `severity_index` correlates 0.903 with `delta_T` (looks redundant) | Despite high pairwise correlation, it encodes a genuine multiplicative interaction effect | It's simply a rescaled/noisy version of `delta_T` with no independent value | Nested-model partial F-test: R² 0.408→0.438, F=7.57, **p=0.0067** | **High** that it adds real incremental signal, based on the partial F-test rather than the pairwise correlation alone |

---

## 7. Ranked top-5 physics-informed engineered features

Ranked by the combined weight of statistical evidence above (correlation strength across
methods, zero-split effect size, and — where relevant — incremental/partial-test evidence
over correlated alternatives), not by chemical plausibility alone:

1. **`avg_temp = (inlet_temperature_K + jacket_temperature_K) / 2`** — strongest predictor of
   `overall_yield` by *every* correlation method tested (Pearson −0.637, Spearman −0.711,
   Kendall −0.517, MI 0.417, dcor 0.670) and the 2nd-largest zero-split effect size (d=1.61).
   Physical motivation: both k1(T) and k2(T) respond to the effective/resultant temperature
   the reacting fluid experiences, not to either boundary condition alone. Adds no linear
   information beyond having both raw temperatures (§5b) — its value is for tree-based models
   and for making the "average thermal level" axis explicit and interpretable in the pitch.

2. **`residence_proxy = length_m / flow_rate_L_min`** — the cleanest quantitative confirmation
   of the series-reaction τ_opt theory in the whole dataset: statistically null on every
   monotonic-correlation method (Pearson/Spearman/Kendall all non-significant, sign even
   flips) yet 2nd-highest Mutual Information (0.339) of anything tested — the textbook
   signature of a non-monotonic (inverted-U) relationship. Also the cleanest, most consistent
   driver of the zero-yield split (Mann–Whitney p=7.8e-5, 3 orders of magnitude tighter than
   `length_m` alone).

3. **`severity_index = residence_proxy × delta_T_jacket_inlet`** — a proxy Damköhler-type
   interaction term. Despite a concerning 0.903 pairwise correlation with `delta_T` alone,
   survives a nested-model partial F-test with a statistically significant incremental
   contribution (p=0.0067) — the only candidate feature in this registry validated via a
   formal incremental-information test rather than correlation alone.

4. **`delta_T_jacket_inlet = jacket_temperature_K − inlet_temperature_K`** — very large,
   highly significant zero-split effect (Cohen's d=0.85, Welch p=4.9e-6) representing the net
   heating boundary condition. Ranked below `avg_temp`/`severity_index` because its
   standalone continuous-target correlations are comparatively weak (Pearson −0.129 ns,
   Spearman −0.240) — its main demonstrated value so far is discriminating the *zero-yield
   regime specifically*, not predicting magnitude within the active regime.

5. **`inv_inlet_temp = 1 / inlet_temperature_K`** (Arrhenius-style transform) — statistically
   significant (Spearman 0.378, matching `inlet_temperature_K` exactly since it's a monotonic
   transform of it) but adds essentially no new ranking information over the raw
   `inlet_temperature_K` in this narrow 352–499 K range (the 1/T nonlinearity is close to
   linear over such a narrow relative range). Lowest-priority of the five — kept in the
   registry as "Pending" rather than promoted, since Phase 2/4 model comparisons will decide
   whether the literal Arrhenius form earns its keep over the simpler raw term.

Full status tracking for these and any future candidates: [`reports/feature_registry.md`](feature_registry.md).

---

## Next step

Phase 2: build the validated/promising features above into the modeling feature set,
re-run the same correlation/redundancy battery on the finalized set (guarding against
introducing new collinearity), and begin baseline model comparisons (Phase 4) to get the
first real, model-based (not just correlation-based) evidence on which engineered features
actually reduce RMSE.
