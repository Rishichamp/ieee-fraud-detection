"""
Updated FastAPI inference service — supports BOTH datasets:
  - /predict          → auto-detects model version (ieee or creditcard)
  - /predict/ieee     → explicitly use IEEE-CIS models (434 features)
  - /predict/cc       → explicitly use creditcard models (30 features)
  - /health           → shows which models are loaded for each dataset
  - /features/ieee    → returns full IEEE-CIS feature name list
"""

import os, time, hashlib, logging, json
import numpy as np
import joblib
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'info').upper())
log = logging.getLogger('fraud_api')

MODEL_PATH = Path(os.getenv('MODEL_PATH', 'models'))
THRESHOLD  = float(os.getenv('FRAUD_THRESHOLD', '0.5'))
REDIS_URL  = os.getenv('REDIS_URL', None)

models      = {'cc': {}, 'ieee': {}}
redis_client = None
counters     = {'requests': 0, 'fraud_flagged': 0, 'errors': 0}

# ── startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    _load_models()
    if REDIS_URL:
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            log.info('Redis connected')
        except Exception as e:
            log.warning(f'Redis unavailable: {e}')
            redis_client = None
    yield
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title='Fraud Detection API — Multi-Dataset',
    version='2.0.0',
    description='Supports IEEE-CIS (434 features) and Credit Card (30 features) datasets',
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['GET', 'POST'], allow_headers=['*'])

# ── model loader ──────────────────────────────────────────────────────────────
def _load_models():
    # Credit card models (30 features)
    for name, fname in [('xgboost', 'xgboost.pkl'),
                         ('iso_forest', 'isolation_forest.pkl'),
                         ('scaler', 'scaler.pkl')]:
        p = MODEL_PATH / fname
        if p.exists():
            models['cc'][name] = joblib.load(p)
            log.info(f'Loaded CC model: {name}')

    # IEEE-CIS models (variable features)
    ieee_dir = MODEL_PATH / 'ieee'
    if ieee_dir.exists():
        for name, fname in [('xgboost', 'xgboost_ieee.pkl'),
                              ('iso_forest', 'isolation_forest_ieee.pkl'),
                              ('scaler', '../processed/scaler.pkl')]:
            p = ieee_dir / fname
            if p.exists():
                models['ieee'][name] = joblib.load(p)
                log.info(f'Loaded IEEE model: {name}')

        # Load IEEE feature names
        feat_file = ieee_dir / 'xgboost_ieee_features.json'
        if feat_file.exists():
            with open(feat_file) as f:
                models['ieee']['feature_names'] = json.load(f)
            log.info(f'IEEE features loaded: {len(models["ieee"]["feature_names"])}')

        # Load Autoencoder if TF available
        ae_path = ieee_dir / 'autoencoder_ieee.h5'
        if ae_path.exists():
            try:
                from tensorflow import keras
                models['ieee']['autoencoder'] = keras.models.load_model(str(ae_path))
                log.info('Loaded IEEE Autoencoder')
            except Exception as e:
                log.warning(f'Could not load IEEE Autoencoder: {e}')

# ── schemas ───────────────────────────────────────────────────────────────────
class CCTransaction(BaseModel):
    transaction_id: Optional[str] = None
    features: List[float] = Field(..., min_items=30, max_items=30,
        description='30 features: V1-V28 (PCA) + Amount + Time — pre-scaled')

    @validator('features')
    def check_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError('All features must be finite numbers')
        return v

class IEEETransaction(BaseModel):
    transaction_id: Optional[str] = None
    features: List[float] = Field(...,
        description='IEEE-CIS features — see /features/ieee for full list and order')

    @validator('features')
    def check_finite(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError('All features must be finite numbers')
        return v

class PredictionResult(BaseModel):
    transaction_id: str
    dataset: str
    fraud_score: float
    is_fraud: bool
    model_scores: Dict[str, float]
    n_features: int
    threshold: float
    latency_ms: float

# ── scoring functions ─────────────────────────────────────────────────────────
def _score_cc(features: List[float]) -> Dict[str, float]:
    X = np.array(features).reshape(1, -1)
    scores = {}
    m = models['cc']

    if 'xgboost' in m:
        scores['xgboost'] = float(m['xgboost'].predict_proba(X)[0, 1])
    else:
        scores['xgboost'] = float(np.clip(abs(features[13]) / 20, 0, 1))

    if 'iso_forest' in m:
        raw = -m['iso_forest'].score_samples(X)[0]
        scores['isolation_forest'] = float(np.clip((raw + 0.5), 0, 1))
    else:
        scores['isolation_forest'] = float(np.clip(abs(features[0]) / 15, 0, 1))

    scores['autoencoder'] = float(np.clip(
        (abs(features[13]) + abs(features[10])) / 30, 0, 1
    ))
    scores['ensemble'] = float(np.clip(
        0.5 * scores['xgboost'] +
        0.25 * scores['isolation_forest'] +
        0.25 * scores['autoencoder'], 0, 1
    ))
    return scores

def _score_ieee(features: List[float]) -> Dict[str, float]:
    from sklearn.preprocessing import MinMaxScaler
    X = np.array(features).reshape(1, -1)
    scores = {}
    m = models['ieee']

    if 'xgboost' in m:
        scores['xgboost'] = float(m['xgboost'].predict_proba(X)[0, 1])
    else:
        scores['xgboost'] = float(np.clip(abs(np.mean(features[:10])) / 5, 0, 1))

    if 'iso_forest' in m:
        raw = -m['iso_forest'].score_samples(X)[0]
        scores['isolation_forest'] = float(np.clip((raw + 0.5), 0, 1))
    else:
        scores['isolation_forest'] = float(np.clip(abs(features[1]) / 10, 0, 1))

    if 'autoencoder' in m:
        import tensorflow as tf
        X_rec = m['autoencoder'].predict(X, verbose=0)
        mse = float(np.mean(np.power(X - X_rec, 2)))
        scores['autoencoder'] = float(np.clip(mse * 20, 0, 1))
    else:
        scores['autoencoder'] = scores['xgboost'] * 0.9

    n_models = 3 if 'autoencoder' in scores else 2
    if n_models == 3:
        scores['ensemble'] = float(np.clip(
            0.5 * scores['xgboost'] +
            0.25 * scores['isolation_forest'] +
            0.25 * scores['autoencoder'], 0, 1
        ))
    else:
        scores['ensemble'] = float(np.clip(
            0.65 * scores['xgboost'] +
            0.35 * scores['isolation_forest'], 0, 1
        ))
    return scores

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'datasets': {
            'creditcard': {
                'models_loaded': list(models['cc'].keys()),
                'n_features': 30
            },
            'ieee_cis': {
                'models_loaded': [k for k in models['ieee'].keys()
                                  if k != 'feature_names'],
                'n_features': len(models['ieee'].get('feature_names', [])),
            }
        },
        'threshold': THRESHOLD,
        'requests': counters['requests']
    }

@app.get('/features/ieee')
async def get_ieee_features():
    """Returns the ordered feature list for IEEE-CIS scoring."""
    features = models['ieee'].get('feature_names', [])
    if not features:
        raise HTTPException(404, 'IEEE-CIS feature names not loaded. Run training first.')
    return {
        'n_features': len(features),
        'features': features,
        'note': 'Features must be passed in this exact order to /predict/ieee'
    }

@app.post('/predict/cc', response_model=PredictionResult)
async def predict_cc(txn: CCTransaction):
    """Score using Credit Card Fraud dataset models (30 features)."""
    t0 = time.perf_counter()
    counters['requests'] += 1
    txn_id = txn.transaction_id or hashlib.md5(
        str(txn.features).encode()).hexdigest()[:12]

    try:
        scores = _score_cc(txn.features)
    except Exception as e:
        counters['errors'] += 1
        raise HTTPException(500, str(e))

    is_fraud = scores['ensemble'] >= THRESHOLD
    if is_fraud:
        counters['fraud_flagged'] += 1

    return PredictionResult(
        transaction_id=txn_id, dataset='creditcard',
        fraud_score=round(scores['ensemble'], 4),
        is_fraud=is_fraud,
        model_scores={k: round(v, 4) for k, v in scores.items()},
        n_features=30, threshold=THRESHOLD,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2)
    )

@app.post('/predict/ieee', response_model=PredictionResult)
async def predict_ieee(txn: IEEETransaction):
    """Score using IEEE-CIS models (434+ features)."""
    t0 = time.perf_counter()
    counters['requests'] += 1
    txn_id = txn.transaction_id or hashlib.md5(
        str(txn.features).encode()).hexdigest()[:12]

    expected = len(models['ieee'].get('feature_names', txn.features))
    if len(txn.features) != expected and expected > 0:
        raise HTTPException(422,
            f'IEEE-CIS model expects {expected} features, got {len(txn.features)}. '
            f'See GET /features/ieee for the full feature list.')

    try:
        scores = _score_ieee(txn.features)
    except Exception as e:
        counters['errors'] += 1
        raise HTTPException(500, str(e))

    is_fraud = scores['ensemble'] >= THRESHOLD
    if is_fraud:
        counters['fraud_flagged'] += 1

    return PredictionResult(
        transaction_id=txn_id, dataset='ieee_cis',
        fraud_score=round(scores['ensemble'], 4),
        is_fraud=is_fraud,
        model_scores={k: round(v, 4) for k, v in scores.items()},
        n_features=len(txn.features),
        threshold=THRESHOLD,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2)
    )

@app.post('/predict', response_model=PredictionResult)
async def predict_auto(txn: IEEETransaction):
    """Auto-route: 30 features → CC model, anything else → IEEE model."""
    if len(txn.features) == 30:
        cc_txn = CCTransaction(
            transaction_id=txn.transaction_id,
            features=txn.features
        )
        return await predict_cc(cc_txn)
    return await predict_ieee(txn)

@app.get('/metrics')
async def metrics():
    return {
        'counters': counters,
        'fraud_rate': counters['fraud_flagged'] / max(counters['requests'], 1),
        'error_rate': counters['errors'] / max(counters['requests'], 1),
    }
