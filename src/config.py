"""
config.py
Central place for file paths and column names.
Keeping them here means we change a name ONCE and every file stays in sync.
"""
from pathlib import Path

# Folder paths (relative to the project root, no matter where you run Python from)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATA_PATH = DATA_DIR / "students_raw.csv"        # messy data (as we "collected" it)
CLEAN_DATA_PATH = DATA_DIR / "students_clean.csv"    # data after cleaning
MODEL_PATH = MODELS_DIR / "student_performance_model.joblib"
METRICS_PATH = MODELS_DIR / "model_comparison.csv"

# Column names
TARGET = "result"   # "Pass" / "Fail" in the raw data, 1 / 0 after cleaning

NUMERIC_FEATURES = [
    "study_hours",                # hours studied per day
    "attendance",                 # attendance percentage (0-100)
    "previous_score",             # previous exam score (0-100)
    "assignment_score",           # 0-100
    "internal_score",             # internal assessment, 0-100
    "sleep_hours",                # hours of sleep per night
    "previous_failed_subjects",   # count of subjects failed before
]
CATEGORICAL_FEATURES = ["class_participation"]          # Low / Medium / High
PARTICIPATION_ORDER = ["Low", "Medium", "High"]

# Features we create ourselves (see features.py)
ENGINEERED_FEATURES = ["avg_academic_score", "has_failed_before", "low_attendance"]

# Valid ranges, used to detect incorrect values during cleaning
VALID_RANGES = {
    "study_hours": (0, 16),
    "attendance": (0, 100),
    "previous_score": (0, 100),
    "assignment_score": (0, 100),
    "internal_score": (0, 100),
    "sleep_hours": (2, 14),
    "previous_failed_subjects": (0, 10),
}

RANDOM_STATE = 42   # makes results repeatable
