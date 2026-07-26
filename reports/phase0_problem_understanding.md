# Phase 0 — Problem Understanding & Reactor Domain Reasoning

## 1. The physical system

The introduction states the data comes from **CFD/BVP simulations** of a **non-isothermal,
continuous-flow reactor**. The only geometric feature we get is `length_m`, with no
diameter/cross-section — this is consistent with a 1-D **plug flow reactor (PFR)** model,
where a spatial coordinate `z` runs along the reactor length and two coupled ODEs are
integrated along it:

- a **species balance** for A, B, C
- an **energy balance** for local temperature T(z), coupled to the jacket via a heat-transfer
  term ∝ `(T_jacket − T(z))`

Calling this a "BVP" (rather than a pure IVP) strongly suggests the energy balance is solved
with a boundary condition tied to the jacket, which is exactly why `jacket_temperature_K`
cannot be treated as just another independent linear predictor — it's a *boundary condition*
that reshapes the entire temperature trajectory the reaction experiences.

## 2. The reaction network

```
A --k1--> B --k2--> C      (both Arrhenius: k_i(T) = A_i * exp(-E_i / R T))
```

This is the textbook **series (consecutive) reaction** case. For first-order kinetics at a
fixed space time τ (batch or PFR equivalent):

```
C_A(τ) = C_A0 * exp(-k1 τ)
C_B(τ) = C_A0 * k1/(k2 - k1) * [exp(-k1 τ) − exp(-k2 τ)]
```

Two structural facts fall out of this that should dominate our feature engineering:

1. **C_B has an interior maximum in τ** (residence time), at
   `τ_opt = ln(k2/k1) / (k2 − k1)`. Too short → A hasn't converted yet (yield low because
   B hasn't formed). Too long → B has already converted onward to C (yield low because
   B has been consumed). **Yield vs. residence time is non-monotonic — a plain linear
   term cannot represent it; we need τ and τ² (or better, τ interacted with a temperature
   term) as engineered features.**
2. **As τ → ∞, C_B → 0** regardless of starting concentration — over-reaction drives the
   desired product to extinction. This is a strong physical candidate explanation for
   observed exact-zero yields (see §4).

Because both k1 and k2 rise with temperature but (in a typical hackathon design) the side
reaction's activation energy E2 is often set higher than E1, raising temperature is a
double-edged sword: it speeds up the desired conversion *and* accelerates the very side
reaction that destroys the product once formed. **This selectivity/conversion trade-off is
the "hidden nonlinear relationship" the problem statement explicitly flags.**

## 3. What each raw feature actually controls physically

| Feature | Physical role | Modeling implication |
|---|---|---|
| `flow_rate_L_min` | Sets residence/space time: τ ∝ V/Q. With `length_m` fixing V (up to a constant cross-section), τ ∝ `length_m / flow_rate_L_min`. | Should not be used only as a raw linear term — its *ratio* with length is the physically meaningful quantity. |
| `concentration_mol_L` (C_A0) | Inlet concentration of A. For ideal first-order series kinetics, the **yield fraction C_B/C_A0 is independent of C_A0** — it only matters via non-idealities, or via the **adiabatic temperature rise** it drives in an exothermic reaction (more concentrated feed → more heat released per unit volume → higher local T → shifts the k1/k2 balance). Hypothesis: C_A0 mostly acts *indirectly*, through the energy balance, not directly through kinetics. Testable in Phase 1/8. |
| `inlet_temperature_K` | Starting point of the temperature trajectory before jacket effects. | Baseline driver of both k1(T) and k2(T) at reactor entrance. |
| `length_m` | Sets space time jointly with flow rate; also sets how much *contact time* the fluid has with the jacket (more heat-transfer opportunity). | Appears in both the kinetic (τ) and thermal (heat-exchange area/time) roles — likely needs to appear in more than one engineered feature. |
| `jacket_temperature_K` | External boundary condition. The **sign and magnitude of `jacket_temperature_K − inlet_temperature_K`** determines whether the reactor net-heats or net-cools along its length. | Expect this differential (not the raw jacket temperature) to be far more predictive than either temperature alone. |

## 4. The zero-yield cluster — a first empirical look

25% of the training rows (37/150) have **exactly** `overall_yield = 0.0`. A quick directional
check (not the full Phase-1 EDA, just enough to sanity-check the hypothesis before committing
engineering effort) compares those 37 rows against the other 113:

| Quantity | Zero-yield avg | Non-zero avg |
|---|---|---|
| `flow_rate_L_min` | 30.12 | 43.85 |
| `concentration_mol_L` | 2.15 | 2.36 |
| `inlet_temperature_K` | 439.0 | 419.7 |
| `length_m` | 15.81 | 13.52 |
| `jacket_temperature_K` | 499.2 | 423.6 |
| residence proxy `length_m/flow_rate_L_min` | 0.758 | 0.505 |
| **`jacket_temperature_K − inlet_temperature_K`** | **+60.2** | **+3.9** |

This lines up cleanly with the theory in §2: the zero-yield rows combine **longer residence
time** with **much stronger net heating from the jacket** (+60 K vs +4 K gap) and higher
inlet temperature — exactly the "long τ + high T" corner that drives k2-dominated
over-reaction, converting essentially all of B onward into C. This is *not* proof (that's
what formal Phase-1 EDA + Phase-9 residual analysis are for), but it's a strong enough signal
to shape the plan:

- The yield surface likely has a **sharp collapse region**, not just a smooth trade-off — a
  single homogeneous regressor may underfit that boundary. **Hypothesis to test in Phase 4/5:
  a two-stage model (classify "extinguished" vs. regress magnitude given "active") may
  outperform a single regressor**, at the cost of added complexity and an extra source of
  error compounding at the boundary.
- Any residence-time and net-thermal-driving-force features designed in Phase 2 should be
  evaluated specifically for how well they separate this cluster, not just for overall
  correlation with yield.

## 5. Alternatives considered for framing the problem

| Framing | Pros | Cons | Decision |
|---|---|---|---|
| Plain regression on raw 5 features | Simplest, fully general (trees can learn interactions) | Ignores known physics, wastes the ~150-sample budget re-learning τ-optimum and thermal coupling from scratch, hard to defend to judges as "understanding the system" | Reject as sole approach — used only as a naive baseline for comparison |
| Regression on raw + physics-derived features (τ, ΔT, Damköhler-like groupings) | Injects known structure, likely to need far fewer samples to fit well, directly defensible in the pitch | Requires assumptions (e.g., first-order kinetics, fixed cross-section) that may not exactly match the hidden simulator | **Primary approach** |
| Two-stage hurdle model (classifier for zero vs. regressor for magnitude) | Directly targets the observed zero-inflation structure | Adds a second model/boundary source of error; only worth it if it beats a single model in CV | Evaluate in Phase 4/5, keep only if it wins |
| Full mechanistic re-derivation (fit k1(T), k2(T), solve ODEs ourselves) | Maximum physical grounding | We don't know true rate laws/activation energies/reactor cross-section; likely to overfit wrong mechanistic assumptions with only 150 points | Reject as primary method, but use its *functional forms* (Arrhenius, τ-optimum) as inspiration for engineered features in Phase 2 |

## 6. Expected benefit / complexity of physics-informed features (preview of Phase 2)

- **High expected benefit, low complexity**: `length_m / flow_rate_L_min` (residence proxy),
  `jacket_temperature_K − inlet_temperature_K` (net thermal driving force). Both are one-line
  derived ratios/differences directly motivated by §2–§4.
- **Medium benefit, medium complexity**: interaction terms between residence proxy and an
  average/effective temperature; polynomial (quadratic) residence term to capture the
  interior-maximum shape.
- **Speculative, worth testing but not assuming**: Arrhenius-style `exp(-1/T)` transforms of
  inlet/jacket temperature (without knowing true activation energies these are just flexible
  nonlinear bases, not literal physics); an explicit "reaction severity index" combining τ and
  ΔT multiplicatively as a proxy Damköhler number.

## Next step

Phase 1: full quantitative EDA (distributions, correlations — Pearson/Spearman/Kendall,
mutual information, duplicate/outlier checks) on both `train_dataset.csv` and
`test_dataset.csv`, explicitly testing the hypotheses above rather than exploring blindly.
