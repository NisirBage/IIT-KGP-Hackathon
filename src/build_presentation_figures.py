"""Two summary figures built specifically for the Phase 8 presentation deck, from numbers
already established in prior phases (no new modeling) -- a leaderboard bar chart and an
ensemble-vs-single-model comparison with confidence intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "phase8"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Leaderboard (Phase 4 baseline RMSE, all 13 models)
# ---------------------------------------------------------------------------
with open(REPORTS / "phase4_analysis_results.json") as f:
    phase4 = json.load(f)
lb = sorted(phase4["leaderboard"], key=lambda r: r["rmse_mean"])
names = [r["model"] for r in lb]
means = [r["rmse_mean"] for r in lb]
stds = [r["rmse_std"] for r in lb]
colors = ["#C44E52" if n == "ExtraTrees" else "#4C72B0" for n in names]

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = np.arange(len(names))
ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("RMSE (RepeatedKFold 5x10)")
ax.set_title("Phase 4 Baseline Leaderboard — 13 Models, Identical Protocol")
plt.tight_layout()
plt.savefig(FIG_DIR / "leaderboard.png", dpi=140)
plt.close(fig)
print("Saved leaderboard.png")

# ---------------------------------------------------------------------------
# Figure 2: Ensemble improvement -- ExtraTrees alone vs. final blend, with CI
# ---------------------------------------------------------------------------
with open(REPORTS / "phase6_final_blend_results.json") as f:
    blend = json.load(f)

et_mean, et_std = 16.693, 2.235  # Phase 4 baseline, RepeatedKFold(5,10)
blend_mean, blend_std = blend["loo_clipped"]["mean"], blend["loo_clipped"]["std"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
labels = ["ExtraTrees\n(single model)", "3-Model Blend\n(ET+CB+RF, clipped)"]
vals = [et_mean, blend_mean]
errs = [et_std, blend_std]
bar_colors = ["#4C72B0", "#55A868"]
ax.bar(labels, vals, yerr=errs, color=bar_colors, alpha=0.85, capsize=6, width=0.55)
ax.set_ylabel("RMSE")
ax.set_title("Final Model vs. Best Single Model")
for i, (v, e) in enumerate(zip(vals, errs)):
    ax.text(i, v + e + 0.3, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")

ax = axes[1]
region_rows = blend["region_specific"]
regions = [r["region"].replace("_", " ") for r in region_rows]
et_r = [r["extratrees_rmse"] for r in region_rows]
bl_r = [r["blend_rmse"] for r in region_rows]
x = np.arange(len(regions))
width = 0.35
ax.bar(x - width/2, et_r, width, label="ExtraTrees", color="#4C72B0", alpha=0.85)
ax.bar(x + width/2, bl_r, width, label="Blend", color="#55A868", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=15, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Improvement Is Broad-Based, Not Concentrated")
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "ensemble_improvement.png", dpi=140)
plt.close(fig)
print("Saved ensemble_improvement.png")
