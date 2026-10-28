"""Like-for-like comparison of three models on the SAME 97 URL features and the
SAME grouped split used by train_url_model.py (duplicate feature rows never
appear in both train and test). Does not overwrite any saved model.

Run:  python Implementation\\compare_url_models.py
Out:  results/url_only_model_comparison.csv
"""
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "dataset_full.csv"
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="phishing").iloc[:, :97].copy()
    y = df["phishing"].copy()
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        .split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train rows: {len(train_idx):,} | Test rows: {len(test_idx):,}")

    models = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42,
                                           n_jobs=-1))]),
        "Linear SVM": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LinearSVC(random_state=42, max_iter=5000))]),
    }

    rows = []
    for name, model in models.items():
        started = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = (model.predict_proba(X_test)[:, 1]
                 if hasattr(model, "predict_proba")
                 else model.decision_function(X_test))
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, score),
            "training_seconds": round(time.time() - started, 2)})
        print(f"done: {name}")

    out = pd.DataFrame(rows).round(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "url_only_model_comparison.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()