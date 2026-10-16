"""
features.py
1) add_features()      -> creates new columns from existing ones (feature engineering)
2) build_pipeline()    -> ONE object that does: add features -> fill gaps -> scale/encode -> model

Because the whole thing is a single Pipeline, the exact same steps run during
training and inside the Streamlit app. That is how we avoid training/serving mismatch.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, PARTICIPATION_ORDER,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new, more informative columns. Works row by row, so it cannot leak."""
    df = df.copy()

    # Average of the three academic scores (mean ignores missing values)
    df["avg_academic_score"] = df[["previous_score", "assignment_score", "internal_score"]].mean(axis=1)

    # 1 if the student has failed any subject before, else 0 (keep NaN if unknown)
    df["has_failed_before"] = (df["previous_failed_subjects"] > 0).astype(float)
    df.loc[df["previous_failed_subjects"].isna(), "has_failed_before"] = np.nan

    # 1 if attendance is below 75% (a common minimum requirement), else 0
    df["low_attendance"] = (df["attendance"] < 75).astype(float)
    df.loc[df["attendance"].isna(), "low_attendance"] = np.nan
    return df


ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES
ALL_FEATURES = ALL_NUMERIC + CATEGORICAL_FEATURES


def build_pipeline(model) -> Pipeline:
    """Wrap any sklearn classifier in the full preprocessing pipeline."""
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),   # median is robust to outliers
        ("scale", StandardScaler()),                          # same scale for every number
    ])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        # Low < Medium < High has a natural order, so we encode it as 0, 1, 2
        ("encode", OrdinalEncoder(categories=[PARTICIPATION_ORDER])),
    ])
    preprocessor = ColumnTransformer([
        ("numbers", numeric_steps, ALL_NUMERIC),
        ("categories", categorical_steps, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("add_features", FunctionTransformer(add_features)),
        ("preprocess", preprocessor),
        ("model", model),
    ])
