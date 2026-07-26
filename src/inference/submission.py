"""Writes the competition submission CSV in the exact required format:
exactly 50 rows, exactly one column named `overall_yield`, no index, UTF-8, float values
rounded to >=3 decimal places, row order identical to the input test CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def write_submission(predictions: np.ndarray, out_path: Path | None = None) -> Path:
    out_path = out_path or (config.SUBMISSION_DIR / f"{config.TEAM_NAME}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({config.TARGET_COLUMN: predictions})
    # float_format="%.3f" forces a fixed 3-decimal STRING representation (e.g. "0.000",
    # "69.040") -- np.round() alone rounds the underlying float correctly but pandas'
    # default CSV writer drops trailing zeros ("0.0", "69.04"), which does not literally
    # display "at least 3 decimal places" as the competition spec requires. Found during
    # the Phase 9 adversarial audit: 25/50 rows in the original submission displayed fewer
    # than 3 decimal digits despite being numerically correct -- a real risk if the
    # competition platform does a literal string-format check.
    df.to_csv(out_path, index=False, encoding="utf-8", float_format="%.3f")
    return out_path
