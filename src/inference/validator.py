"""Input, prediction, and submission-format validation. Every function here raises a
descriptive exception on failure -- this pipeline fails loudly, never silently produces
a questionable prediction (Core Principle: fail loudly rather than silently produce
incorrect predictions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class ValidationError(Exception):
    """Raised whenever input, prediction, or submission validation fails."""


def validate_input_schema(df: pd.DataFrame) -> dict:
    """Column names, order, dtypes, missing values, duplicates, row count."""
    report = {"checks": [], "passed": True}

    def check(name: str, condition: bool, detail: str):
        report["checks"].append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            report["passed"] = False

    actual_cols = list(df.columns)
    missing_cols = [c for c in config.EXPECTED_COLUMNS if c not in actual_cols]
    unexpected_cols = [c for c in actual_cols if c not in config.EXPECTED_COLUMNS]
    check("no_missing_columns", len(missing_cols) == 0, f"missing={missing_cols}")
    check("no_unexpected_columns", len(unexpected_cols) == 0, f"unexpected={unexpected_cols}")
    check("column_order", actual_cols == config.EXPECTED_COLUMNS,
          f"expected={config.EXPECTED_COLUMNS} actual={actual_cols}")

    if missing_cols:
        raise ValidationError(f"Input schema validation failed: missing required columns {missing_cols}")
    if unexpected_cols:
        raise ValidationError(f"Input schema validation failed: unexpected columns {unexpected_cols}")

    for col in config.EXPECTED_COLUMNS:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        check(f"dtype_{col}", is_numeric, f"dtype={df[col].dtype}")
        if not is_numeric:
            raise ValidationError(f"Input schema validation failed: column '{col}' is not numeric (dtype={df[col].dtype})")

    n_missing = int(df[config.EXPECTED_COLUMNS].isnull().sum().sum())
    check("no_missing_values", n_missing == 0, f"n_missing_values={n_missing}")
    if n_missing > 0:
        raise ValidationError(f"Input schema validation failed: {n_missing} missing values in input data")

    n_dupes = int(df.duplicated().sum())
    check("no_duplicate_rows", n_dupes == 0, f"n_duplicate_rows={n_dupes}")
    # duplicates are suspicious but not necessarily fatal -- warn via report, don't raise

    check("row_count", len(df) == config.EXPECTED_ROW_COUNT, f"expected={config.EXPECTED_ROW_COUNT} actual={len(df)}")
    if len(df) != config.EXPECTED_ROW_COUNT:
        raise ValidationError(f"Input schema validation failed: expected {config.EXPECTED_ROW_COUNT} rows, got {len(df)}")

    return report


def validate_predictions(preds: np.ndarray, n_expected: int) -> dict:
    """NaN, infinite, count, range, clipping stats, duplicate predictions."""
    report = {"checks": [], "passed": True}

    def check(name: str, condition: bool, detail):
        report["checks"].append({"check": name, "passed": bool(condition), "detail": str(detail)})
        if not condition:
            report["passed"] = False

    check("prediction_count", len(preds) == n_expected, f"expected={n_expected} actual={len(preds)}")
    if len(preds) != n_expected:
        raise ValidationError(f"Prediction validation failed: expected {n_expected} predictions, got {len(preds)}")

    n_nan = int(np.isnan(preds).sum())
    check("no_nan", n_nan == 0, f"n_nan={n_nan}")
    if n_nan > 0:
        raise ValidationError(f"Prediction validation failed: {n_nan} NaN predictions")

    n_inf = int(np.isinf(preds).sum())
    check("no_inf", n_inf == 0, f"n_inf={n_inf}")
    if n_inf > 0:
        raise ValidationError(f"Prediction validation failed: {n_inf} infinite predictions")

    in_range = bool(np.all((preds >= config.PREDICTION_MIN) & (preds <= config.PREDICTION_MAX)))
    check("within_physical_bounds", in_range,
          f"range=[{preds.min():.4f},{preds.max():.4f}] expected=[{config.PREDICTION_MIN},{config.PREDICTION_MAX}]")
    if not in_range:
        raise ValidationError(f"Prediction validation failed: predictions outside [{config.PREDICTION_MIN},{config.PREDICTION_MAX}]")

    n_unique = len(np.unique(preds))
    check("duplicate_predictions_flag", True, f"n_unique={n_unique}/{len(preds)} (informational, not fatal)")

    report["summary_stats"] = {
        "mean": float(preds.mean()), "std": float(preds.std()),
        "min": float(preds.min()), "max": float(preds.max()),
        "median": float(np.median(preds)),
        "n_at_lower_bound": int((preds == config.PREDICTION_MIN).sum()),
        "n_at_upper_bound": int((preds == config.PREDICTION_MAX).sum()),
    }
    return report


def validate_submission_file(path, expected_rows: int = config.EXPECTED_ROW_COUNT) -> dict:
    """Post-write validation of the actual CSV artifact: shape, header, dtype, encoding."""
    report = {"checks": [], "passed": True}

    def check(name: str, condition: bool, detail):
        report["checks"].append({"check": name, "passed": bool(condition), "detail": str(detail)})
        if not condition:
            report["passed"] = False

    raw_bytes = path.read_bytes()
    try:
        raw_bytes.decode("utf-8")
        is_utf8 = True
    except UnicodeDecodeError:
        is_utf8 = False
    check("utf8_encoding", is_utf8, "decodable as UTF-8")

    df = pd.read_csv(path)
    check("row_count", len(df) == expected_rows, f"expected={expected_rows} actual={len(df)}")
    check("column_count", df.shape[1] == 1, f"expected=1 actual={df.shape[1]}")
    check("header_name", list(df.columns) == [config.TARGET_COLUMN], f"expected=['{config.TARGET_COLUMN}'] actual={list(df.columns)}")
    check("no_index_column", not str(df.columns[0]).strip().lower() in ("", "unnamed: 0", "index"),
          f"first_column='{df.columns[0]}'")
    # Guard: only inspect dtype of the target column if it actually exists under the
    # expected name -- otherwise this check itself would crash with a raw KeyError instead
    # of failing cleanly through the report (caught and fixed during Phase 7 testing).
    if config.TARGET_COLUMN in df.columns:
        target_series = df[config.TARGET_COLUMN]
        check("float_values", pd.api.types.is_numeric_dtype(target_series), f"dtype={target_series.dtype}")
    else:
        check("float_values", False, f"column '{config.TARGET_COLUMN}' not present, cannot check dtype")

    if not report["passed"]:
        failed = [c["check"] for c in report["checks"] if not c["passed"]]
        raise ValidationError(f"Submission file validation failed: {failed}")
    return report
