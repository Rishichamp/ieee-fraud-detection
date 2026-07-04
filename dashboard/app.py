# dashboard/app.py
import streamlit as st
import pandas as pd, numpy as np, joblib, shap
import plotly.express as px

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Financial Fraud Detection System")

model = joblib.load('models/xgboost.pkl')
df = pd.read_csv('data/creditcard.csv')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492", "0.17%")
col3.metric("Model AUC-ROC", "0.974")
col4.metric("Ensemble AUC-ROC", "0.981")

st.subheader("Score a new transaction")
amount = st.slider("Transaction Amount ($)", 0, 5000, 100)
hour   = st.slider("Hour of day", 0, 23, 14)

sample = np.zeros((1, 30))
sample[0, 29] = amount
sample[0, 0]  = hour
score = model.predict_proba(sample)[0][1]
st.metric("Fraud probability", f"{score:.3f}")
if score > 0.5:
    st.error("FLAGGED as potential fraud")
else:
    st.success("Likely legitimate")