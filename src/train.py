"""
train.py
Loads data -> cleans -> splits -> trains 3 models -> evaluates -> saves the best.
Run from the project root:  python -m src.train
"""
import joblib
import matplotlib
matplotlib.use("Agg")   # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CLEAN_DATA_PATH, FIGURES_DIR, METRICS_PATH, MODEL_PATH, MODELS_DIR,
    RANDOM_STATE, RAW_DATA_PATH, TARGET,
)
from src.data_cleaning import clean_data
from src.features import ALL_FEATURES, build_pipeline


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and clean --------------------------------------------------
    df = clean_data(pd.read_csv(RAW_DATA_PATH))
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # Raw feature columns (the Pipeline creates the engineered ones itself)
    raw_feature_columns = [c for c in ALL_FEATURES if c in df.columns]
    X = df[raw_feature_columns]
    y = df[TARGET]

    # ---- 2. Train-test split BEFORE any learning step -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,            # 20% of students are kept hidden for testing
        random_state=RANDOM_STATE,  # same split every run
        stratify=y,               # keep the Pass/Fail ratio the same in both parts
    )
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # ---- 3. The three beginner-friendly models ------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        pipeline = build_pipeline(model)

        # Cross-validation on TRAINING data only: a fair score used to pick the winner.
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1").mean()

        pipeline.fit(X_train, y_train)          # learn from training data
        predictions = pipeline.predict(X_test)  # predict on unseen students
        fitted[name] = pipeline

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1-score": f1_score(y_test, predictions),
            "CV F1 (train)": cv_f1,
        })

    results_table = pd.DataFrame(results).set_index("Model").round(3)
    print("\n=== Model comparison (test set) ===")
    print(results_table.to_string())

    # ---- 4. Pick the best model --------------------------------------------
    # We choose by cross-validated F1 on the TRAINING data, so the test set stays
    # a truly untouched final exam.
    best_name = results_table["CV F1 (train)"].idxmax()
    best_pipeline = fitted[best_name]
    print(f"\nBest model: {best_name}")

    best_predictions = best_pipeline.predict(X_test)
    print("\nConfusion matrix [rows = actual, columns = predicted] (0=Fail, 1=Pass):")
    print(confusion_matrix(y_test, best_predictions))
    print("\nDetailed report:")
    print(classification_report(y_test, best_predictions, target_names=["Fail", "Pass"]))

    # Save a picture of the confusion matrix for every model
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pipe.predict(X_test), display_labels=["Fail", "Pass"],
            cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=120)
    plt.close()

    # ---- 5. Save ------------------------------------------------------------
    results_table.to_csv(METRICS_PATH)
    joblib.dump({"pipeline": best_pipeline, "model_name": best_name,
                 "metrics": results_table.loc[best_name].to_dict(),
                 "feature_columns": raw_feature_columns}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
