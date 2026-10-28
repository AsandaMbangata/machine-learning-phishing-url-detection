"""Runs a labelled URL set through the model and saves real results.

Edit URLS with links you KNOW the label of, then:  python tests\\run_evidence.py
Output: results/url_validation_results.csv (use in report Appendix A + slides).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Implementation.url_feature_extractor import extract_url_features  # noqa: E402

# (url, true_label)  0 = legitimate, 1 = phishing  -- ADD MORE, ~20+ each
URLS = [
    ("https://www.google.com", 0),
    ("https://example.com/login", 0),
    ("https://en.wikipedia.org/wiki/Phishing", 0),
    ("http://secure-login-paypal.verify-account.xyz/update?id=1", 1),
]

model = joblib.load(ROOT / "models" / "random_forest_url_only.joblib")
names = json.loads((ROOT / "models" / "url_feature_names.json").read_text())
rows = []
for url, truth in URLS:
    p = model.predict_proba(pd.DataFrame([extract_url_features(url)])[names])[0][1]
    pred = int(p >= 0.5)
    rows.append({"url": url, "true": truth, "pred": pred,
                 "p_phishing": round(p, 3), "result": "PASS" if pred == truth else "FAIL"})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "url_validation_results.csv", index=False)
print(df.to_string(index=False))
print("Correct:", (df.result == "PASS").sum(), "/", len(df))
