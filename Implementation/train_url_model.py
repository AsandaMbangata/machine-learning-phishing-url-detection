
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "dataset_full.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    df = pd.read_csv(DATA_PATH)

    # First 97 features are derived from URL structure.
    # The remaining 14 require external/domain information.
    target_index = df.columns.get_loc("phishing")
    X = df.drop(columns="phishing").iloc[:, :97].copy()
    y = df["phishing"].copy()

    if X.shape[1] != 97:
        raise ValueError(f"Expected 97 features, got {X.shape[1]}")

    # Keep identical feature vectors in the same split to reduce
    # optimistic evaluation caused by duplicate feature rows.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training URL-only Random Forest...")
    print(f"Training rows: {len(train_idx):,}")
    print(f"Test rows: {len(test_idx):,}")
    print(f"URL-derived features: {X.shape[1]}")
    started = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "URL-only Random Forest",
        "features": int(X.shape[1]),
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
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
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "training_seconds": round(time.time() - started, 2),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "random_forest_url_only.joblib")

    with open(
        MODEL_DIR / "url_feature_names.json", "w", encoding="utf-8"
    ) as f:
        json.dump(X.columns.tolist(), f, indent=2)

    with open(
        RESULTS_DIR / "url_only_model_metrics.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual legitimate (0)", "Actual phishing (1)"],
        columns=["Predicted legitimate (0)", "Predicted phishing (1)"],
    ).to_csv(RESULTS_DIR / "url_only_confusion_matrix.csv")

    print("\nURL-only model results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nSaved model:", MODEL_DIR / "random_forest_url_only.joblib")
    print("Saved metrics:", RESULTS_DIR / "url_only_model_metrics.json")
    print(
        "\nThis model uses URL-derived features only. "
        "A matching URL feature extractor is still required in the app."
    )


if __name__ == "__main__":
    main()
