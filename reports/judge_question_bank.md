# Judge Question Bank

65 anticipated questions across 5 categories plus cross-cutting "gotcha" questions. Each is answered in detail in
[`technical_defense.md`](technical_defense.md) — this document is the question index;
cross-reference by number.

## Chemical Engineering (14)

1. Why should residence time be non-monotonic with respect to yield?
2. Why does temperature reduce yield in some regimes but not others?
3. Why not solve the governing ODEs/BVP directly instead of using ML?
4. How do you know this is really a series reaction (A→B→C) and not something else?
5. What is `avg_temp` supposed to represent physically, and why does the average of two
   boundary temperatures matter more than either alone?
6. Why does `concentration_mol_L` have zero effect on yield? Isn't that chemically
   surprising?
7. What is the `severity_index` supposed to represent, and is Damköhler number framing
   actually justified here?
8. Why did you choose `exp(-1000/T)` for the Arrhenius-inspired feature? Where does 1000
   come from?
9. How do you know the reactor is really a plug-flow reactor (PFR) and not a CSTR or
   something else?
10. What happens physically at the ~410–480K "transition region," and how confident are you
    in that exact range?
11. Could the zero-yield rows represent a solver failure/numerical artifact rather than true
    physical reactor extinction?
12. If E2 (side reaction activation energy) is actually lower than E1, does your whole
    narrative fall apart?
13. Why does `flow_rate_L_min` matter more through `residence_proxy` than on its own?
14. How would your model behave physically at operating conditions outside the training
    range (extrapolation)?

## Machine Learning (16)

15. Why ExtraTrees as the primary base model instead of Random Forest or a boosting method?
16. Why not deep learning / a neural network?
17. Why not a transformer?
18. Why an ensemble at all, given your own diversity analysis initially suggested limited
    ensemble potential?
19. Why does your ensemble include a negative coefficient on RandomForest? Isn't that a red
    flag?
20. Why is CatBoost tuned but ExtraTrees is not (i.e. left at defaults)?
21. Why only 4 shortlisted model families in Phase 4→5, not all 13 benchmarked models?
22. Why exclude NGBoost and EBM — performance or convenience?
23. Why Optuna over grid search or random search?
24. Why did you reduce the Optuna trial count from the suggested 100–200 down to 25–80?
25. How do you know 51–80 completed trials is "enough" for TPE to converge?
26. Why GaussianProcess in the shortlist at all, given it's the weakest performer and least
    physically plausible?
27. Why drop GaussianProcess from the final ensemble after including it in the search?
28. Why a linear blend instead of a more sophisticated stacking architecture (e.g. a
    gradient-boosted meta-model)?
29. Isn't 150 training rows far too small for reliable ML at all?
30. Why clip predictions to [0,100] instead of using a model that inherently respects the
    bound (e.g. a beta regression or logit-transformed target)?

## Statistics (13)

31. Why should we trust that your reported improvements are statistically meaningful and not
    noise?
32. Why RepeatedKFold(5,10) specifically, and not a different fold count or repeat count?
33. How do you know your hyperparameter tuning didn't overfit the validation protocol?
34. Why did nested CV show the "naive" estimate was worse (not better) than the honest
    estimate — isn't that backwards from the textbook optimistic-bias story?
35. Why did ExtraTrees get statistically worse after tuning? Doesn't that undermine
    confidence in your whole tuning framework?
36. What's your actual confidence interval on the final RMSE, and how was it computed?
37. Why use Friedman/Nemenyi instead of just comparing mean RMSE across models?
38. Isn't testing many features/models/hyperparameters an implicit multiple-comparisons
    problem that inflates your apparent significance?
39. How many of your "statistically significant" findings would survive a Bonferroni or
    similar correction across the whole project?
40. Why leave-one-repeat-out instead of a completely fresh, independent holdout set?
41. Your ensemble's coefficients were fit on the same 150 rows used throughout the entire
    project — how do you know they'll hold on genuinely new data?
42. What is your p-value threshold, and is it pre-registered or chosen after seeing results?
43. Why does Mutual Information disagree with Pearson/Spearman correlation for some
    features, and which should we trust?

## Software Engineering (10)

44. How is your pipeline reproducible? What guarantees determinism?
45. How is inference validated before a prediction is ever produced?
46. How do you prevent data leakage across your whole pipeline, not just within one model?
47. What happens if the test data schema doesn't match what your pipeline expects?
48. How do you know your serialized model artifacts aren't corrupted?
49. What's your process for freezing/versioning a model before submission?
50. If a bug were found in your validator (as one was, during Phase 7), how do you know
    there isn't a similar undiscovered bug elsewhere?
51. How would someone else reproduce your exact results from scratch?
52. Why version Optuna study databases and not just the final hyperparameters?
53. What's your actual test coverage — did you only test the happy path?

## Product / Deployment (9)

54. Can this model run in real time for plant optimization?
55. How would this scale to millions of predictions per day?
56. How would you monitor for model drift in a real deployed system?
57. What would you do if the reactor design changed (new geometry, new catalyst)?
58. How would you retrain this model as new plant data arrives?
59. What's the cost/latency tradeoff of your ensemble vs. a single model in production?
60. How would you communicate prediction uncertainty to a plant operator?
61. What safety guardrails would you add before trusting this model's output in a live
    control loop?
62. What's the very first thing you'd want to know from the plant before deploying this?

## Cross-cutting / "gotcha" questions likely from a skeptical panel (3)

63. Of everything in this project, what are you personally least confident about?
64. If you had to remove one phase of this project as unnecessary complexity, which would
    it be, and why didn't you?
65. Why are 42% of your test predictions exactly zero, and how do you know that's correct
    behavior rather than a bug?
