"""
eda.py
Data exploration (on RAW data) + Exploratory Data Analysis charts (on CLEAN data).
Run from the project root:  python -m src.eda
Charts are saved into reports/figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    CLEAN_DATA_PATH, FIGURES_DIR, NUMERIC_FEATURES, RAW_DATA_PATH, TARGET,
)
from src.data_cleaning import clean_data

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)         # show all columns when printing
pd.set_option("display.max_columns", 20)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=120)
    plt.close()


def explore_raw_data():
    """Step 4: first look at the messy data."""
    df = pd.read_csv(RAW_DATA_PATH)
    print("First 5 rows:\n", df.head(), "\n")
    print("Shape (rows, columns):", df.shape, "\n")
    print("Data types:\n", df.dtypes, "\n")
    print("Missing values per column:\n", df.isnull().sum(), "\n")
    print("Duplicate rows:", df.duplicated().sum(), "\n")
    print("Descriptive statistics:\n", df.describe().round(2), "\n")
    print("Target distribution:\n", df[TARGET].value_counts(), "\n")

    # Chart: missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values()
    plt.figure(figsize=(7, 4))
    missing.plot(kind="barh", color="tomato")
    plt.title("Missing values per column (raw data)")
    plt.xlabel("Number of missing values")
    save("01_missing_values.png")

    # Chart: histograms of every numeric column
    df[NUMERIC_FEATURES].hist(bins=25, figsize=(12, 8), color="steelblue", edgecolor="white")
    plt.suptitle("Distribution of each numeric feature (raw data)")
    save("02_feature_histograms.png")


def eda_charts():
    """Step 6: relationships between features and the result."""
    df = clean_data(pd.read_csv(RAW_DATA_PATH), verbose=False)
    df["result_label"] = df[TARGET].map({1: "Pass", 0: "Fail"})
    palette = {"Pass": "seagreen", "Fail": "indianred"}

    # Pass / Fail distribution
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="result_label", hue="result_label", palette=palette,
                       order=["Fail", "Pass"], legend=False)
    ax.bar_label(ax.containers[0])
    plt.title("Pass / Fail distribution")
    plt.xlabel("")
    save("03_pass_fail_distribution.png")

    # Study hours, attendance, previous score vs result (box plots)
    for number, (col, title) in enumerate([
        ("study_hours", "Study hours vs result"),
        ("attendance", "Attendance vs result"),
        ("previous_score", "Previous exam score vs result"),
    ], start=4):
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="result_label", y=col, hue="result_label", palette=palette,
                    order=["Fail", "Pass"], legend=False)
        plt.title(title)
        plt.xlabel("")
        save(f"{number:02d}_{col}_vs_result.png")

    # Class participation vs pass rate
    pass_rate = df.groupby("class_participation")[TARGET].mean().reindex(["Low", "Medium", "High"]) * 100
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=pass_rate.index, y=pass_rate.values, hue=pass_rate.index,
                     palette="Blues", legend=False)
    ax.bar_label(ax.containers[0], fmt="%.0f%%")
    plt.title("Pass rate by class participation")
    plt.ylabel("Pass rate (%)")
    save("07_participation_vs_pass_rate.png")

    # Previous failed subjects vs pass rate
    pass_rate_failed = df.groupby("previous_failed_subjects")[TARGET].mean() * 100
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=pass_rate_failed.index.astype(int), y=pass_rate_failed.values,
                     hue=pass_rate_failed.index.astype(int), palette="Reds_r", legend=False)
    ax.bar_label(ax.containers[0], fmt="%.0f%%")
    plt.title("Pass rate by number of previously failed subjects")
    plt.xlabel("Previous failed subjects")
    plt.ylabel("Pass rate (%)")
    save("08_failed_subjects_vs_pass_rate.png")

    # Correlation heatmap
    plt.figure(figsize=(9, 7))
    correlation = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (1 = Pass, 0 = Fail)")
    save("09_correlation_heatmap.png")

    print("Correlation of each feature with passing:")
    print(correlation[TARGET].drop(TARGET).sort_values(ascending=False).round(2))


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    explore_raw_data()
    eda_charts()
    print("\nCharts saved to", FIGURES_DIR)
