"""Orchestrates the full, deterministic, no-manual-intervention inference workflow:

    load test CSV -> validate input schema -> predict (feature engineering + preprocessing
    happen INSIDE the loaded pipeline, never here) -> clip -> validate predictions ->
    write submission CSV -> validate the written file.

Returns a structured report dict covering every validation step, so a single call produces
everything needed for the inference report / audit trail.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config
from .loader import load_model, load_test_data
from .submission import write_submission
from .validator import validate_input_schema, validate_predictions, validate_submission_file


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_inference(model_path: Path | None = None, test_data_path: Path | None = None,
                   out_path: Path | None = None) -> dict:
    report = {"steps": []}

    model = load_model(model_path)
    report["steps"].append({"step": "load_model", "path": str(model_path or config.FINAL_MODEL_PATH)})

    df = load_test_data(test_data_path)
    report["steps"].append({"step": "load_test_data", "path": str(test_data_path or config.TEST_DATA_PATH), "n_rows": len(df)})

    input_validation = validate_input_schema(df)
    report["input_validation"] = input_validation
    report["steps"].append({"step": "validate_input_schema", "passed": input_validation["passed"]})

    raw_preds = model.predict(df[config.EXPECTED_COLUMNS])
    report["steps"].append({"step": "predict", "n_predictions": len(raw_preds)})

    # The loaded LinearBlendEnsemble already clips internally (Phase 6) -- clip again here
    # explicitly and unconditionally, so the inference pipeline's correctness does not
    # silently depend on that internal detail of one specific model class.
    clipped_preds = np.clip(raw_preds, config.PREDICTION_MIN, config.PREDICTION_MAX)
    n_clipped = int(np.sum(clipped_preds != raw_preds))
    report["steps"].append({"step": "clip", "n_values_changed_by_clip": n_clipped})

    pred_validation = validate_predictions(clipped_preds, n_expected=len(df))
    report["prediction_validation"] = pred_validation
    report["steps"].append({"step": "validate_predictions", "passed": pred_validation["passed"]})

    out_path = write_submission(clipped_preds, out_path)
    report["steps"].append({"step": "write_submission", "path": str(out_path)})

    submission_validation = validate_submission_file(out_path, expected_rows=len(df))
    report["submission_validation"] = submission_validation
    report["steps"].append({"step": "validate_submission_file", "passed": submission_validation["passed"]})

    report["output_path"] = str(out_path)
    report["output_sha256"] = sha256_of_file(out_path)
    report["predictions"] = clipped_preds.tolist()
    report["all_passed"] = all(
        report[k]["passed"] for k in ("input_validation", "prediction_validation", "submission_validation")
    )
    return report
