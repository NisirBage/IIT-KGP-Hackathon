"""Data loading utilities for the reactor yield hackathon project."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_COLS = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
TARGET_COL = "overall_yield"


def load_train() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "train_dataset.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "test_dataset.csv")
