"""Frozen inference configuration. Phase 7: the model architecture is locked -- this file
points at the exact artifact selected in Phase 6 and nothing here should change without a
critical-bug justification (see submission_pipeline_report.md).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_MODEL_PATH = PROJECT_ROOT / "artifacts" / "tuned_pipelines" / "FINAL_ENSEMBLE_blend_v1.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test_dataset.csv"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TEAM_NAME = "TeamName"  # placeholder -- replace with the actual team name before final upload

# Expected raw input schema (Phase 0-1) -- order matters, this is what the frozen pipeline expects
EXPECTED_COLUMNS = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
EXPECTED_ROW_COUNT = 50
EXPECTED_DTYPE_KIND = "f"  # numpy dtype.kind for float

TARGET_COLUMN = "overall_yield"
PREDICTION_MIN = 0.0
PREDICTION_MAX = 100.0
