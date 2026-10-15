# 🛡️ Financial Fraud Detection System — IEEE-CIS Dataset

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready **financial fraud detection system** built on the IEEE-CIS Fraud Detection dataset. Combines four machine learning models — XGBoost, Isolation Forest, Autoencoder, and Graph GNN — into an ensemble with real-time SHAP explainability, deployed via Docker microservices.

---

## 📊 Results

| Model | AUC-ROC | Notes |
|---|---|---|
| XGBoost | **0.9589** | Supervised, best single model |
| Isolation Forest | 0.7689 | Unsupervised anomaly detection |
| Autoencoder | ~0.887 | Deep learning reconstruction |
| **Ensemble** | **0.9543** | Weighted combination |

Dataset: **590,540 transactions · 3.5% fraud rate · 448 features**

---

## 🏗️ Architecture

```
Internet
    │
    ▼
┌─────────┐
│  Nginx  │  :443 — SSL + rate limiting (30 req/min)
└────┬────┘
     ├────────────────────┐
     ▼                    ▼
┌──────────┐       ┌───────────┐
│ FastAPI  │       │Streamlit  │
│  :8000   │       │  :8501    │
│ /predict │       │ Dashboard │
└────┬─────┘       └─────┬─────┘
     └──────────┬─────────┘
                ▼
         ┌────────────┐
         │   Redis    │
         │   :6379    │
         │ Score cache│
         └────────────┘
                │
                ▼
      ┌──────────────────┐
      │  model_store vol │
      │  xgboost.pkl     │
      │  iso_forest.pkl  │
      │  autoencoder.h5  │
      └──────────────────┘
```

---

## 📁 Project Structure

```
ieee_fraud_detection/
├── api/
│   ├── __init__.py
│   └── main.py              ← FastAPI: /predict, /predict/ieee, /health
├── dashboard/
│   └── app.py               ← Streamlit analytics UI
├── preprocessing/
│   ├── preprocess.py        ← Credit card dataset preprocessing
│   └── ieee_preprocess.py   ← IEEE-CIS preprocessing pipeline
├── scripts/
│   ├── train_ieee.py        ← Train all models on IEEE-CIS
│   ├── train_all.py         ← Train all models on creditcard.csv
│   └── deploy.sh            ← One-command deployment
├── models/
│   ├── xgboost_model.py     ← XGBoost training module
│   ├── isolation_forest.py  ← Isolation Forest module
│   ├── autoencoder.py       ← Autoencoder module
│   ├── graph_model.py       ← Graph detection module
│   └── ieee/                ← Trained model artifacts
│       ├── xgboost_ieee.pkl
│       ├── isolation_forest_ieee.pkl
│       ├── xgboost_ieee_features.json
│       └── registry_ieee.json
├── explainability/
│   └── shap_explain.py      ← SHAP global + local explanations
├── nginx/
│   └── nginx.conf           ← Reverse proxy configuration
├── data/
│   ├── ieee_cis/            ← IEEE-CIS CSVs (download separately)
│   └── creditcard/          ← Credit card CSV (download separately)
├── Dockerfile               ← Dashboard container
├── Dockerfile.api           ← API container
├── docker-compose.yml       ← Full stack definition
├── docker-compose.ieee.yml  ← IEEE-CIS override
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ieee-fraud-detection.git
cd ieee-fraud-detection
```

### 2. Get the IEEE-CIS dataset
```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials: https://www.kaggle.com/settings/api
# Accept rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules

kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis
cd data/ieee_cis && unzip ieee-fraud-detection.zip && cd ../..
```

### 3. Set up environment
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 4. Preprocess + Train
```bash
python preprocessing/ieee_preprocess.py   # ~5-10 min
python scripts/train_ieee.py              # ~15-20 min
```

### 5. Run locally
```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/app.py --server.port 8501
```

### 6. Or run with Docker
```bash
make ssl
make build
make up
```

---

## 🔌 API Reference

### Health check
```bash
curl http://localhost:8000/health
```

### Score a transaction (IEEE-CIS model)
```bash
curl -X POST http://localhost:8000/predict/ieee \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TXN-001", "features": [0.12, -0.53, 1.07, ...]}'
```
Features go in the order of `GET /features/ieee`, standardised like the
training data (`data/ieee_cis/processed`). To send raw encoded values instead,
add `"prescaled": false`; the server then applies the training
`StandardScaler`, which must exist at `SCALER_PATH`
(default `data/ieee_cis/processed/scaler.pkl`, written by
`preprocessing/ieee_preprocess.py`).

`model_scores` only contains models that are actually loaded (the committed
artifacts are XGBoost + Isolation Forest; the autoencoder joins when
`models/ieee/autoencoder_ieee.keras` and TensorFlow are present). A dataset
with no trained model returns `503` rather than a guessed score.

### Get feature list
```bash
curl http://localhost:8000/features/ieee
```

### Batch scoring
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"features": [...]}, {"features": [...]}]}'
```
Up to 1000 transactions per request; returns a list of results.

### Response format
```json
{
  "transaction_id": "TXN-001",
  "dataset": "ieee_cis",
  "fraud_score": 0.9412,
  "is_fraud": true,
  "model_scores": {
    "xgboost": 0.9610,
    "isolation_forest": 0.8830,
    "ensemble": 0.9412
  },
  "n_features": 448,
  "threshold": 0.5,
  "latency_ms": 4.2
}
```

---

## 🧠 Models

### XGBoost (AUC: 0.9589)
- 500 trees, max_depth=7, learning_rate=0.05
- SMOTE oversampling (fraud: 3.5% → 9.1%)
- scale_pos_weight=27 for class imbalance
- Early stopping on validation AUC

### Isolation Forest (AUC: 0.7689)
- 300 estimators, contamination=0.035
- Trained on legitimate transactions only
- Catches novel fraud patterns unseen in training

### Autoencoder
- Architecture: 448→256→128→64→16→64→128→256→448
- Trained on legitimate transactions only
- Reconstruction MSE used as anomaly score

### Graph Detection
- NetworkX DiGraph: card1 nodes → addr1 nodes
- 13,886 nodes, 39,974 edges
- Degree centrality and fraud ring detection

---

## 📊 IEEE-CIS Feature Groups

| Group | Features | Description |
|---|---|---|
| Transaction | TransactionAmt, ProductCD | Amount and product type |
| Card | card1-card6 | Card number, type, bank, CVV |
| Address | addr1, addr2 | Billing address |
| Email | P_emaildomain, R_emaildomain | Purchaser and recipient email |
| Count | C1-C14 | Linked addresses, cards, emails |
| Timedelta | D1-D15 | Days between transaction events |
| Match | M1-M9 | Name, address, CVV match flags |
| Vesta V | V1-V339 | Engineered anonymous features |
| Identity | id_01-id_38 | Device, browser, OS info |
| Engineered | hour, log_amount, etc. | Derived in preprocessing |

---

## 🐳 Docker Services

| Service | Image | Port | Role |
|---|---|---|---|
| nginx | nginx:1.25-alpine | 80/443 | Reverse proxy, SSL, rate limiting |
| api | fraud_detection-api | 8000 | FastAPI ML inference |
| dashboard | fraud_detection-dashboard | 8501 | Streamlit UI |
| redis | redis:7-alpine | 6379 | Score cache, rate limiter |
| trainer | fraud_detection-api | — | One-shot model training |

---

## 📋 Requirements

```
Python 3.10+
xgboost>=2.0
scikit-learn>=1.3
imbalanced-learn
pandas, numpy, joblib
fastapi, uvicorn
streamlit, plotly
shap, networkx
redis
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔮 Future Roadmap

- [ ] Real-time Kafka streaming pipeline
- [ ] PyTorch Geometric GNN implementation
- [ ] Online learning with River library
- [ ] Federated learning across institutions
- [ ] Prometheus + Grafana monitoring
- [ ] EU AI Act compliance audit
- [ ] Model drift detection with Evidently AI
