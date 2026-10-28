
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "dataset_full.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "phishing"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    # Use numeric engineered features only; never include the target as input.
    X = df.drop(columns=[TARGET]).select_dtypes(include="number")
    y = df[TARGET]

    if X.empty:
        raise ValueError("No numeric feature columns were found.")
    if y.isna().any():
        raise ValueError("The target contains missing values.")
    if y.nunique() != 2:
        raise ValueError(
            f"Expected a binary target, but found labels: {y.unique().tolist()}"
        )

    print(f"Rows: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Target counts: {y.value_counts().to_dict()}")
    print("Note: label 1 is treated as the positive class for metrics.")
    print("Verify the dataset documentation to confirm what each label means.")

    # Fixed held-out test set; stratification preserves label proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                random_state=RANDOM_STATE,
                max_iter=5000,
            )),
        ]),
    }

    all_results = []

    for name, pipeline in models.items():
        safe_name = name.lower().replace(" ", "_")
        print(f"\nTraining {name}...")
        started = time.time()

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # All three selected estimators provide a continuous decision score
        # or probability score suitable for ROC-AUC.
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "training_seconds": round(time.time() - started, 2),
        }

        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.joblib")

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ).to_csv(RESULTS_DIR / f"{safe_name}_confusion_matrix.csv")

        with open(
            RESULTS_DIR / f"{safe_name}_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as report_file:
            report_file.write(classification_report(
                y_test, predictions, zero_division=0
            ))

        all_results.append(metrics)

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"Time:      {metrics['training_seconds']:.2f} seconds")
        print(f"Saved model: {MODEL_DIR / (safe_name + '.joblib')}")

    results = pd.DataFrame(all_results)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with open(
        RESULTS_DIR / "model_comparison.json",
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(all_results, results_file, indent=2)

    print("\n=== Model comparison (held-out test set) ===")
    print(results.round(4).to_string(index=False))
    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Models saved in:  {MODEL_DIR}")
    print(
        "\nImportant: these models expect the dataset's engineered features, "
        "not a raw URL string. The app is not connected to them yet."
    )


if __name__ == "__main__":
    main()
