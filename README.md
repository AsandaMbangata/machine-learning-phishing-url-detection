# machine-learning-phishing-url-detection

Capstone prototype (CPT580S) that estimates whether a URL looks like phishing using only the structure of the URL. The website itself is never opened.

The app combines two things:
- **Transparent rule-based checks** (HTTPS, IP host, `@` sign, length, subdomains, keywords, URL shorteners).
- **A Random Forest** trained on 97 URL-derived features, shown as an estimated phishing probability.

## Dataset
`data/dataset_full.csv`: phishing website dataset with 88,647 URLs and 111 features plus a `phishing` label (0 = legitimate, 1 = phishing). Only the first 97 features are derived from the URL string; the remaining 14 need live DNS, WHOIS or search lookups and are not used.
Absent URL components (no directory, file or query) are encoded as `-1` in the dataset, and the feature extractor follows this convention.

## Setup (Windows, Python 3.12)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Create the model
Model files (`*.joblib`) are not stored in the repository because of their size. Generate them from the dataset:
```powershell
python Implementation\train_url_model.py
```
This trains the URL-only Random Forest and writes `models/random_forest_url_only.joblib`, `models/url_feature_names.json`, and metrics to `results/`.
`Implementation\train_models.py` trains and compares the three baseline models (Logistic Regression, Random Forest, Linear SVM).

## Run the app
```powershell
streamlit run app.py
```
Open http://localhost:8501 and paste a URL.

## Run the tests
```powershell
python -m pytest tests -v
python tests\run_evidence.py
```
`tests/run_evidence.py` scores a labelled list of URLs and saves the results to `results/url_validation_results.csv`.

## Results
URL-only Random Forest (the model used by the app), 97 features, held-out test set of 16,766 rows (identical feature vectors kept in the same split):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.928 | 0.902 | 0.903 | 0.903 | 0.976 |

Baseline comparison (`results/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.931 | 0.890 | 0.915 | 0.902 | 0.979 |
| Random Forest | 0.971 | 0.955 | 0.960 | 0.958 | 0.995 |
| Linear SVM | 0.930 | 0.883 | 0.919 | 0.900 | 0.979 |

Automated tests: 12 of 12 passing (`python -m pytest tests -v`).

## Limitations
- The model sees URL structure only. It cannot see page content or a site's reputation, so it can be wrong in both directions.
- It can produce false positives on short URLs with a single path: in the training data, 89.7% of URLs with the same shape as `example.com/login` are phishing (defect D-003).
- Feature extraction rules were inferred from patterns in the dataset, not from the original generator code, so an exact match cannot be claimed.
- The output is an estimate and must not be the only basis for deciding whether a site is safe.

See `docs/Defect_Log.md` for defects found and fixed during testing.