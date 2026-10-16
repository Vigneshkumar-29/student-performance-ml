"""
generate_data.py
Creates a realistic, deliberately MESSY student dataset so we can practise
cleaning it. Run from the project root:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, RANDOM_STATE, DATA_DIR

N_STUDENTS = 1500


def create_students(n=N_STUDENTS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    # A hidden "ability" trait makes the columns realistically related to each other
    ability = rng.normal(0, 1, n)

    study_hours = np.clip(4 + 1.2 * ability + rng.normal(0, 1.5, n), 0, 12)
    attendance = np.clip(78 + 8 * ability + rng.normal(0, 8, n), 35, 100)
    previous_score = np.clip(62 + 12 * ability + rng.normal(0, 8, n), 10, 100)
    assignment_score = np.clip(66 + 10 * ability + rng.normal(0, 9, n), 10, 100)
    internal_score = np.clip(64 + 11 * ability + rng.normal(0, 9, n), 10, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.1, n), 3.5, 10)

    # Participation: better students tend to participate more
    p_high = 1 / (1 + np.exp(-(ability - 0.4) * 1.5))
    p_low = 1 / (1 + np.exp((ability + 0.6) * 1.5))
    roll = rng.random(n)
    participation = np.where(roll < p_low, "Low",
                             np.where(roll > 1 - p_high * 0.8, "High", "Medium"))

    # Weaker students have failed more subjects before
    failed_rate = np.clip(1.0 - 0.7 * ability, 0.05, 4)
    previous_failed = np.clip(rng.poisson(failed_rate), 0, 6)

    # Pass / Fail is decided by a probability built from the features
    participation_points = pd.Series(participation).map(
        {"Low": -0.6, "Medium": 0.0, "High": 0.6}).values
    score = (
        0.07 * (study_hours - 4)
        + 0.03 * (attendance - 75)
        + 0.05 * (previous_score - 60)
        + 0.03 * (assignment_score - 65)
        + 0.05 * (internal_score - 62)
        + 0.25 * (sleep_hours - 6.8)
        + participation_points
        - 0.55 * previous_failed
        + 0.9
    )
    pass_probability = 1 / (1 + np.exp(-score))
    result = np.where(rng.random(n) < pass_probability, "Pass", "Fail")

    return pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous_score.round(1),
        "assignment_score": assignment_score.round(1),
        "internal_score": internal_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "class_participation": participation,
        "previous_failed_subjects": previous_failed.astype(float),
        "result": result,
    })


def make_messy(df, seed=RANDOM_STATE):
    """Add the kinds of problems real data has."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # 1) Missing values (about 3.5% in several columns)
    for col in ["study_hours", "attendance", "previous_score", "assignment_score", "internal_score",
                "sleep_hours", "class_participation", "previous_failed_subjects"]:
        rows = rng.choice(n, size=int(0.035 * n), replace=False)
        df.loc[rows, col] = np.nan

    # 2) Incorrect values (typing mistakes / sensor errors)
    df.loc[rng.choice(n, 6, replace=False), "attendance"] = 120.0       # > 100 %
    df.loc[rng.choice(n, 5, replace=False), "study_hours"] = -3.0       # negative
    df.loc[rng.choice(n, 4, replace=False), "sleep_hours"] = 24.0       # impossible
    df.loc[rng.choice(n, 5, replace=False), "internal_score"] = 150.0   # > 100

    # 3) Inconsistent text in a categorical column
    rows = rng.choice(n, 40, replace=False)
    df.loc[rows[:15], "class_participation"] = "high"
    df.loc[rows[15:30], "class_participation"] = "  Medium "
    df.loc[rows[30:], "class_participation"] = "LOW"

    # 4) Duplicate rows (same student entered twice)
    duplicates = df.sample(25, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    messy = make_messy(create_students())
    messy.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(messy)} rows to {RAW_DATA_PATH}")
