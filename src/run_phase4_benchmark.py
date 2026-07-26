"""
Phase 4: Baseline Model Benchmarking.

Runs every model in models.MODEL_REGISTRY through an identical RepeatedKFold(5,10) CV
(models.configs), checkpointing each model's full result to
reports/phase4_raw/<model>.json immediately after it finishes -- so an interrupted run
(background execution proved unreliable for long jobs in Phase 3) loses at most one
in-progress model, not the whole benchmark. Re-running this script skips any model whose
checkpoint already exists.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import TARGET_COL, load_train  # noqa: E402
from models import MODEL_REGISTRY, run_repeated_cv, summarize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "reports" / "phase4_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

train = load_train()
X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].values

only = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())

for model_name in only:
    ckpt = RAW_DIR / f"{model_name}.json"
    if ckpt.exists():
        print(f"[skip] {model_name} already checkpointed")
        continue
    print(f"[run] {model_name} ...", flush=True)
    t0 = time.perf_counter()
    try:
        result = run_repeated_cv(model_name, X, y)
    except Exception as e:
        print(f"[FAIL] {model_name}: {e}", flush=True)
        with open(ckpt, "w") as f:
            json.dump({"model": model_name, "error": str(e)}, f, indent=2)
        continue
    elapsed = time.perf_counter() - t0
    summary = summarize(result)
    payload = {
        "model": model_name,
        "family": MODEL_REGISTRY[model_name]["family"],
        "fold_rows": result["fold_rows"],
        "oof_predictions": result["oof_predictions"].tolist(),
        "fit_time_mean": result["fit_time_mean"], "fit_time_std": result.get("fit_time_std"),
        "predict_time_mean": result["predict_time_mean"],
        "summary": summary,
        "total_wall_time_sec": elapsed,
    }
    with open(ckpt, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[done] {model_name}  RMSE={summary['rmse_mean']:.3f}+/-{summary['rmse_std']:.3f}  wall={elapsed:.1f}s", flush=True)

print("Benchmark pass complete.")
