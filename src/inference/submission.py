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
    df = pd.DataFrame({config.TARGET_COLUMN: np.round(predictions, 3)})
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
