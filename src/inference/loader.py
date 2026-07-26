"""Loads the frozen ensemble artifact and raw test data. No preprocessing happens here --
that lives entirely inside the loaded pipeline object (Core Principle 2: zero manual
preprocessing outside the pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Importing models.ensemble registers LinearBlendEnsemble with joblib's unpickler.
from models.ensemble import LinearBlendEnsemble  # noqa: F401,E402

from . import config


def load_model(path: Path | None = None):
    path = path or config.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Cannot proceed with inference.")
    model = joblib.load(path)
    return model


def load_test_data(path: Path | None = None) -> pd.DataFrame:
    path = path or config.TEST_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found at {path}.")
    return pd.read_csv(path)
