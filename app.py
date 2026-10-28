
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse
import json
import joblib
import pandas as pd

from Implementation.url_feature_extractor import extract_url_features

import streamlit as st

# ---------- Paths and configuration ----------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_url_only.joblib"
FEATURES_PATH = PROJECT_DIR / "models" / "url_feature_names.json"

@st.cache_resource
def load_url_model():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if len(feature_names) != 97:
        raise ValueError("The model must use exactly 97 features.")

    return model, feature_names

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f0f6ff 0%, #ffffff 48%);
        color: #172b46;
    }

    [data-testid="stHeader"] {
        background: rgba(240, 246, 255, 0.92);
    }

    .hero {
        padding: 30px;
        border: 1px solid #c7daf5;
        border-radius: 18px;
        background: linear-gradient(
            115deg, #102746 0%, #1558a6 72%, #1976ed 100%
        );
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(16, 39, 70, 0.12);
    }

    .hero h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
    }

    .hero p {
        color: #e1edff;
        margin: 8px 0 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e4f3;
        border-radius: 12px;
        padding: 14px;
    }

    div.stFormSubmitButton > button[kind="primary"] {
        background: #1769e0;
        border-color: #1769e0;
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Phishing URL Detection</h1>
        <p>Machine Learning-Based Phishing URL Detection</p>
        <p>
            Analyse URL structure, review warning signs, and assess
            potentially suspicious web addresses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Model status ----------
model_files_found = MODEL_PATH.exists() and FEATURES_PATH.exists()

left, right = st.columns([3, 1])

with left:
    st.subheader("URL Scanner")
    st.write(
        "Enter a website address below to inspect its URL structure. "
        "The website itself will not be opened."
    )

with right:
    st.metric(
        "Model files",
        "Available" if model_files_found else "Missing",
    )
    st.caption("Experimental ML prediction enabled.")

st.info(
    "The URL-only Random Forest analyses the structure of the URL and shows an estimated phishing probability, alongside transparent rule-based checks. The website itself is never opened"
)


# ---------- URL rule analysis ----------
def inspect_url(raw_url):
    value = raw_url.strip()

    if not value:
        return None, "Enter a URL to continue."

    if any(ch.isspace() for ch in value):
        return None, "The URL contains spaces. Check it and try again."

    candidate = (
        value if "://" in value else "https://" + value
    )

    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None, "The URL format is invalid. Check it and try again."

    if not host or "." not in host:
        return None, "That doesn't look like a complete web address."

    checks = []

    # 1. HTTPS scheme
    checks.append({
        "Check": "HTTPS scheme",
        "Finding": (
            "HTTPS present"
            if parsed.scheme == "https"
            else "HTTPS not present"
        ),
        "Warning": parsed.scheme != "https",
    })

    # 2. IP address used as host
    try:
        ipaddress.ip_address(host)
        uses_ip = True
    except ValueError:
        uses_ip = False

    checks.append({
        "Check": "IP address as host",
        "Finding": (
            "Host is an IP address"
            if uses_ip
            else "Host is a domain name"
        ),
        "Warning": uses_ip,
    })

    # 3. At-sign in the address
    has_at = "@" in parsed.netloc
    checks.append({
        "Check": "At-sign in address",
        "Finding": "Contains @" if has_at else "No @ found",
        "Warning": has_at,
    })

    # 4. Internationalised domain indicator
    punycode = "xn--" in host
    checks.append({
        "Check": "Internationalised domain",
        "Finding": (
            "Contains xn-- domain label"
            if punycode
            else "No xn-- label found"
        ),
        "Warning": punycode,
    })

    # 5. URL length
    long_url = len(value) >= 100
    checks.append({
        "Check": "URL length",
        "Finding": (
            f"{len(value)} characters"
            + (" (long)" if long_url else "")
        ),
        "Warning": long_url,
    })

    # 6. Number of subdomains
    subdomain_count = max(0, len(host.split(".")) - 2)
    many_subdomains = subdomain_count >= 3

    checks.append({
        "Check": "Subdomain count",
        "Finding": f"{subdomain_count} subdomain(s)",
        "Warning": many_subdomains,
    })

    # 7. Potentially sensitive keywords
    suspicious_terms = [
        "login",
        "verify",
        "account",
        "password",
        "secure",
        "update",
        "wallet",
        "payment",
        "signin",
    ]

    matched_terms = [
        term
        for term in suspicious_terms
        if re.search(re.escape(term), value, re.IGNORECASE)
    ]

    checks.append({
        "Check": "Potentially sensitive keywords",
        "Finding": (
            ", ".join(matched_terms)
            if matched_terms
            else "No listed keywords found"
        ),
        "Warning": bool(matched_terms),
    })

    # 8. Known URL shorteners
    shorteners = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "is.gd",
        "ow.ly",
        "rb.gy",
        "cutt.ly",
    }

    is_shortener = host in shorteners

    checks.append({
        "Check": "Known URL-shortener domain",
        "Finding": (
            "Shortener domain detected"
            if is_shortener
            else "No listed shortener detected"
        ),
        "Warning": is_shortener,
    })

    warning_count = sum(
        item["Warning"] for item in checks
    )

    if warning_count >= 4:
        level = "Higher warning signs"
    elif warning_count >= 2:
        level = "Some warning signs"
    else:
        level = "Fewer warning signs"

    return {
        "submitted": value,
        "host": host,
        "checks": checks,
        "warning_count": warning_count,
        "level": level,
    }, None

def predict_url(raw_url):
    model, feature_names = load_url_model()
    features = extract_url_features(raw_url)

    if list(features.keys()) != feature_names:
        raise ValueError(
            "Extracted feature names or order do not match the model."
        )

    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    phishing_index = list(model.classes_).index(1)
    phishing_probability = float(probabilities[phishing_index])

    return prediction, phishing_probability

# ---------- Scanner form ----------
with st.form("url_scan_form"):
    entered_url = st.text_input(
        "Website URL",
        placeholder="https://example.com/login",
        help=(
            "Only the URL text is inspected. "
            "The website is not opened."
        ),
    )

    submitted = st.form_submit_button(
        "Analyse URL",
        type="primary",
        use_container_width=True,
    )


# ---------- Results ----------
if submitted:
    result, error = inspect_url(entered_url)

    if error:
        st.error(error)

    else:
        st.divider()
        st.subheader("Scan Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rule warnings",
            f"{result['warning_count']} / 8",
        )
        col2.metric(
            "Rule-based assessment",
            result["level"],
        )
        col3.metric(
            "Domain",
            result["host"],
        )

        if result["warning_count"] >= 4:
            st.error(
                "Several warning signs were found. Treat this URL "
                "cautiously and verify it through a trusted source."
            )

        elif result["warning_count"] >= 2:
            st.warning(
                "Some warning signs were found. Check the domain "
                "carefully before sharing information."
            )

        else:
            st.success(
                "Few warning signs were detected by these rules. "
                "This does not mean the URL is safe."
            )

        st.markdown("### Machine-learning prediction")

        st.warning(
            "Experimental prediction based on URL structure only. It cannot see page content or the site's reputation, so it can be wrong in both directions. Do not rely on it alone to decide whether a site is safe."
        )

        try:
            prediction, probability = predict_url(entered_url)

            ml_col1, ml_col2 = st.columns(2)

            ml_col1.metric(
                "Random Forest result",
                "Potential phishing" if prediction == 1
                else "Potentially legitimate",
            )

            ml_col2.metric(
                "Estimated phishing probability",
                f"{probability:.1%}",
            )

        except Exception as exc:
            st.error(
                "The model could not analyse this URL. "
                f"Details: {exc}"
            )

        st.markdown("### Detailed checks")

        st.dataframe(
            [
                {
                    "Check": item["Check"],
                    "Finding": item["Finding"],
                    "Result": (
                        "Warning"
                        if item["Warning"]
                        else "No warning"
                    ),
                }
                for item in result["checks"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These findings are based on simple URL rules, not a "
            "machine-learning prediction or probability. Phishing URLs "
            "can evade these checks, and legitimate URLs can trigger "
            "warnings. No website was visited."
        )


# ---------- Project information ----------
with st.expander("About this capstone prototype"):
    st.write(
        "The project compares Logistic Regression, Random Forest, "
        "and Linear SVM. A separate URL-only Random Forest was trained "
        "using 97 URL-derived features."
    )

    st.write(
        "The saved URL-only model is not used for live predictions yet. "
        "The feature extractor must be verified against the dataset's "
        "feature definitions, missing-component conventions, and "
        "feature order before predictions are enabled."
    )

    st.write(
        "This prototype does not visit the submitted website, resolve "
        "DNS, inspect certificates, or verify whether a domain is "
        "currently malicious."
    )

st.caption("Phishing URL Detection • Capstone prototype")
