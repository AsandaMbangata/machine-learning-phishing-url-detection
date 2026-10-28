"""Model/integration tests (REQ-004, REQ-008, REQ-009). Needs models/ present."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Implementation.url_feature_extractor import extract_url_features  # noqa: E402

MODEL = ROOT / "models" / "random_forest_url_only.joblib"
NAMES = ROOT / "models" / "url_feature_names.json"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not trained")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL), json.loads(NAMES.read_text())


def p_phish(model, url):
    m, names = model
    row = pd.DataFrame([extract_url_features(url)])[names]
    return m.predict_proba(row)[0][1]


def test_feature_names_match_extractor(model):                # TC-10
    assert model[1] == list(extract_url_features("a.com/x"))


def test_known_legitimate_scores_low(model):                  # TC-11
    assert p_phish(model, "https://www.google.com") < 0.5


def test_suspicious_url_scores_high(model):                   # TC-12
    assert p_phish(model, "http://secure-login-paypal.verify-account.xyz/update?id=1") > 0.5
