# dashboard/app.py
"""
Streamlit analytics dashboard for the Fraud Detection API.
Talks to the FastAPI service over HTTP (API_URL env var) instead of
loading model files directly, so it always reflects whatever the API
actually has loaded.
"""
import os
import random
import numpy as np
import requests
import streamlit as st
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Financial Fraud Detection System")
st.caption(f"Connected to API: {API_URL}")
@st.cache_data(ttl=30)
def get_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()
@st.cache_data(ttl=300)
def get_ieee_features():
    r = requests.get(f"{API_URL}/features/ieee", timeout=5)
    r.raise_for_status()
    return r.json()["features"]
# ── Connection / health check ───────────────────────────────────────────────
try:
    health = get_health()
except Exception as e:
    st.error(f"Could not reach the API at {API_URL}: {e}")
    st.info("Make sure the `api` service is running: `docker compose ps`")
    st.stop()
ieee_info = health["datasets"]["ieee_cis"]
cc_info = health["datasets"]["creditcard"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("API Status", health["status"].upper())
col2.metric("IEEE-CIS Models Loaded", len(ieee_info["models_loaded"]))
col3.metric("IEEE-CIS Features", ieee_info["n_features"])
col4.metric("Fraud Threshold", health["threshold"])

if not ieee_info["models_loaded"]:
    st.warning(
        "No IEEE-CIS models are loaded on the API yet. "
        "Train them first with `docker compose --profile train run --rm trainer`."
    )
    st.stop()

st.divider()
st.subheader("Score a transaction (IEEE-CIS model)")

try:
    feature_names = get_ieee_features()
except Exception as e:
    st.error(f"Could not fetch feature list: {e}")
    st.stop()

st.caption(
    f"The model expects {len(feature_names)} features in a fixed order. "
    "For a quick demo, adjust a few key fields below — the rest are filled "
    "with zeros (neutral values)."
)

# A few human-meaningful fields to expose as inputs; everything else stays 0.
amount = st.slider("Transaction Amount ($)", 0, 5000, 100)
card1 = st.slider("card1 (encoded card ID)", 0, 20000, 5000)
c1 = st.slider("C1 (count feature)", 0, 20, 1)
use_random = st.checkbox("Randomize the remaining features (stress-test)", value=False)

if st.button("Predict fraud probability", type="primary"):
    features = [0.0] * len(feature_names)

    def set_if_present(name, value):
        if name in feature_names:
            features[feature_names.index(name)] = float(value)

    set_if_present("TransactionAmt", amount)
    set_if_present("card1", card1)
    set_if_present("C1", c1)

    if use_random:
        rng = random.Random(42)
        features = [
            v if v != 0.0 else rng.uniform(-1, 1) for v in features
        ]

    payload = {"features": features}

    try:
        resp = requests.post(f"{API_URL}/predict/ieee", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.HTTPError as e:
        st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    st.metric("Fraud probability (ensemble)", f"{result['fraud_score']:.3f}")
    if result["is_fraud"]:
        st.error(f"🚩 FLAGGED as potential fraud (threshold: {result['threshold']})")
    else:
        st.success(f"✅ Likely legitimate (threshold: {result['threshold']})")

    with st.expander("Per-model scores"):
        st.json(result["model_scores"])
    st.caption(f"Latency: {result['latency_ms']:.1f} ms · Transaction ID: {result['transaction_id']}")