"""
data_cleaning.py
Turns the messy raw CSV into a tidy table.

IMPORTANT (data leakage): everything here is a fixed RULE ("remove duplicates",
"scores above 100 are wrong"). Rules don't learn anything from the data, so it
is safe to apply them before the train/test split.
Steps that DO learn from data (filling gaps with the median, scaling) are NOT
done here. They live inside the sklearn Pipeline in features.py and are fitted
on training data only.
"""
import numpy as np
import pandas as pd

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, PARTICIPATION_ORDER, TARGET, VALID_RANGES,
)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    rows_at_start = len(df)

    # 1) Remove duplicate rows (the same student entered twice).
    df = df.drop_duplicates()
    if "student_id" in df.columns:
        df = df.drop_duplicates(subset="student_id")
    if verbose:
        print(f"Duplicates removed: {rows_at_start - len(df)}")

    # 2) Convert data types. Anything that is not a number becomes NaN (missing).
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Fix inconsistent text: "  Medium ", "high", "LOW"  ->  "Medium", "High", "Low"
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip().str.title()
        df.loc[~df[col].isin(PARTICIPATION_ORDER), col] = np.nan   # unknown label -> missing
        # Plain Python strings + np.nan (not pandas' pd.NA), which scikit-learn understands
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    # 4) Incorrect values (outside the possible range) -> mark as missing.
    #    We don't delete the whole row, because the other columns are still useful.
    for col, (low, high) in VALID_RANGES.items():
        wrong = (df[col] < low) | (df[col] > high)
        if verbose and wrong.sum() > 0:
            print(f"Incorrect values in '{col}' set to missing: {int(wrong.sum())}")
        df.loc[wrong, col] = np.nan

    # 5) Target: we cannot learn from a student whose result is unknown -> drop.
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title().map({"Pass": 1, "Fail": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH
    cleaned = clean_data(pd.read_csv(RAW_DATA_PATH))
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved {len(cleaned)} clean rows to {CLEAN_DATA_PATH}")
    print("Missing values still left (the Pipeline will fill them):")
    print(cleaned.isna().sum()[cleaned.isna().sum() > 0])
