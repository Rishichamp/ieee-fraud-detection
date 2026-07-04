"""
IEEE-CIS Fraud Detection — Full Training Pipeline
==================================================
Trains all 4 models on the IEEE-CIS dataset and saves artifacts
to models/ieee/ so they coexist with creditcard.csv models.

Run:
    python scripts/train_ieee.py

Expects:
    data/ieee_cis/processed/X_train.npy etc. (from ieee_preprocess.py)
"""

import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('train_ieee')

DATA_DIR  = Path('data/ieee_cis/processed')
MODEL_DIR = Path('models/ieee')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    log.info('Loading processed IEEE-CIS data...')
    X_train = np.load(DATA_DIR / 'X_train.npy')
    X_test  = np.load(DATA_DIR / 'X_test.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    y_test  = np.load(DATA_DIR / 'y_test.npy')

    with open(DATA_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)

    log.info(f'  Train: {X_train.shape}  Fraud: {y_train.mean()*100:.2f}%')
    log.info(f'  Test:  {X_test.shape}   Fraud: {y_test.mean()*100:.2f}%')
    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# 2. XGBOOST
# =============================================================================
def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    log.info('─' * 50)
    log.info('Training XGBoost on IEEE-CIS...')

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = int(neg / pos)
    log.info(f'  scale_pos_weight = {spw}  (neg={neg:,} / pos={pos:,})')

    try:
        from imblearn.over_sampling import SMOTE
        log.info('  Applying SMOTE...')
        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        log.info(f'  After SMOTE: {X_res.shape}, fraud={y_res.mean()*100:.1f}%')
    except ImportError:
        log.warning('  imbalanced-learn not found — skipping SMOTE')
        X_res, y_res = X_train, y_train

    model = xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 7,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 10,
        scale_pos_weight      = spw,
        eval_metric           = 'auc',
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = 'hist',
    )

    model.fit(
        X_res, y_res,
        eval_set = [(X_test, y_test)],
        verbose  = 50,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)

    log.info(f'  AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}')
    log.info('\n' + classification_report(
        y_test, (y_prob >= 0.5).astype(int), digits=4
    ))

    importance = dict(zip(feature_names, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    log.info('  Top 10 features by importance:')
    for fname, imp in top10:
        log.info(f'    {fname:35s}: {imp:.4f}')

    joblib.dump(model, MODEL_DIR / 'xgboost_ieee.pkl')
    with open(MODEL_DIR / 'xgboost_ieee_features.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    log.info('  Saved → models/ieee/xgboost_ieee.pkl')
    return model, auc


# =============================================================================
# 3. ISOLATION FOREST
# =============================================================================
def train_isolation_forest(X_train, y_train, X_test, y_test):
    log.info('─' * 50)
    log.info('Training Isolation Forest on IEEE-CIS...')

    contamination = float(y_train.mean())
    log.info(f'  contamination = {contamination:.4f}')

    model = IsolationForest(
        n_estimators  = 300,
        contamination = contamination,
        max_samples   = 'auto',
        random_state  = 42,
        n_jobs        = -1,
    )

    X_legit = X_train[y_train == 0]
    log.info(f'  Training on {len(X_legit):,} legitimate transactions...')
    model.fit(X_legit)

    scores = -model.score_samples(X_test)
    auc    = roc_auc_score(y_test, scores)
    log.info(f'  AUC-ROC: {auc:.4f}')

    joblib.dump(model, MODEL_DIR / 'isolation_forest_ieee.pkl')
    log.info('  Saved → models/ieee/isolation_forest_ieee.pkl')
    return model, auc


# =============================================================================
# 4. AUTOENCODER (optional)
# =============================================================================
def train_autoencoder(X_train, y_train, X_test, y_test):
    log.info('─' * 50)
    log.info('Training Autoencoder on IEEE-CIS...')

    try:
        import tensorflow as tf
        from tensorflow import keras
        log.info(f'  TensorFlow version: {tf.__version__}')
    except ImportError:
        log.warning('  TensorFlow not installed — skipping Autoencoder')
        log.warning('  Install with: pip install tensorflow')
        return None, None

    inp_dim = X_train.shape[1]
    X_legit = X_train[y_train == 0]
    log.info(f'  Input dim: {inp_dim}  |  Legit samples: {len(X_legit):,}')

    inp = keras.Input(shape=(inp_dim,))
    x   = keras.layers.Dense(256, activation='relu')(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(128, activation='relu')(x)
    x   = keras.layers.Dropout(0.1)(x)
    x   = keras.layers.Dense(64,  activation='relu')(x)
    x   = keras.layers.Dense(16,  activation='relu')(x)
    x   = keras.layers.Dense(64,  activation='relu')(x)
    x   = keras.layers.Dense(128, activation='relu')(x)
    x   = keras.layers.Dense(256, activation='relu')(x)
    out = keras.layers.Dense(inp_dim, activation='sigmoid')(x)

    autoencoder = keras.Model(inp, out)
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, monitor='val_loss', verbose=1
        ),
    ]

    autoencoder.fit(
        X_legit, X_legit,
        epochs           = 50,
        batch_size       = 512,
        validation_split = 0.1,
        callbacks        = callbacks,
        verbose          = 1,
    )

    X_rec = autoencoder.predict(X_test, batch_size=1024, verbose=0)
    mse   = np.mean(np.power(X_test - X_rec, 2), axis=1)
    auc   = roc_auc_score(y_test, mse)
    log.info(f'  AUC-ROC: {auc:.4f}')

    autoencoder.save(str(MODEL_DIR / 'autoencoder_ieee.keras'))
    log.info('  Saved → models/ieee/autoencoder_ieee.keras')
    return autoencoder, auc


# =============================================================================
# 5. GRAPH DETECTION
# =============================================================================
def build_ieee_graph():
    log.info('─' * 50)
    log.info('Building IEEE-CIS transaction graph...')

    try:
        import networkx as nx
    except ImportError:
        log.warning('  networkx not installed — skipping graph')
        log.warning('  Install with: pip install networkx')
        return None, []

    try:
        txn_path = Path('data/ieee_cis/train_transaction.csv')
        log.info(f'  Reading {txn_path}...')

        txn = pd.read_csv(
            txn_path,
            usecols=['TransactionID', 'card1', 'addr1', 'TransactionAmt', 'isFraud']
        )
        log.info(f'  Loaded {len(txn):,} transactions for graph')

        G = nx.DiGraph()

        log.info('  Building graph edges...')
        for _, row in txn.iterrows():
            card_node = f'CARD_{int(row["card1"])}'
            addr_node = f'ADDR_{int(row["addr1"])}' if pd.notna(row['addr1']) else 'ADDR_UNKNOWN'
            G.add_edge(
                card_node, addr_node,
                txn_id = row['TransactionID'],
                amount = row['TransactionAmt'],
                fraud  = int(row['isFraud'])
            )

        log.info(f'  Graph nodes: {G.number_of_nodes():,}')
        log.info(f'  Graph edges: {G.number_of_edges():,}')

        rings = [c for c in nx.strongly_connected_components(G) if len(c) >= 3]
        log.info(f'  Fraud rings detected (size >= 3): {len(rings)}')

        degree_cent = nx.degree_centrality(G)
        top_cards = sorted(
            [(n, v) for n, v in degree_cent.items() if n.startswith('CARD_')],
            key=lambda x: x[1], reverse=True
        )[:10]

        log.info('  Top 10 cards by degree centrality:')
        for node, cent in top_cards:
            log.info(f'    {node}: {cent:.6f}')

        joblib.dump(
            {'rings': rings, 'top_cards': top_cards},
            MODEL_DIR / 'ieee_graph.pkl'
        )
        log.info('  Saved → models/ieee/ieee_graph.pkl')
        return G, rings

    except Exception as e:
        log.warning(f'  Graph build failed: {e}')
        return None, []


# =============================================================================
# 6. ENSEMBLE EVALUATION
# =============================================================================
def evaluate_ensemble(xgb_model, if_model, ae_model, X_test, y_test):
    log.info('─' * 50)
    log.info('Evaluating ensemble...')

    mms    = MinMaxScaler()
    scores = {}

    scores['xgboost'] = xgb_model.predict_proba(X_test)[:, 1]

    if_raw = -if_model.score_samples(X_test)
    scores['isolation_forest'] = mms.fit_transform(if_raw.reshape(-1, 1)).ravel()

    weights = {'xgboost': 0.5, 'isolation_forest': 0.25}
    w_sum   = 0.75

    if ae_model is not None:
        X_rec  = ae_model.predict(X_test, batch_size=1024, verbose=0)
        ae_mse = np.mean(np.power(X_test - X_rec, 2), axis=1)
        scores['autoencoder'] = mms.fit_transform(ae_mse.reshape(-1, 1)).ravel()
        weights['autoencoder'] = 0.25
        w_sum = 1.0

    ensemble = sum(scores[k] * weights[k] for k in weights) / w_sum

    auc = roc_auc_score(y_test, ensemble)
    ap  = average_precision_score(y_test, ensemble)

    log.info('  Individual AUC-ROC scores:')
    for k, v in scores.items():
        log.info(f'    {k:25s}: {roc_auc_score(y_test, v):.4f}')
    log.info(f'  Ensemble AUC-ROC:       {auc:.4f}')
    log.info(f'  Ensemble Avg Precision: {ap:.4f}')

    return ensemble, auc


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    log.info('=' * 60)
    log.info('IEEE-CIS Fraud Detection — Training Pipeline')
    log.info('=' * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    if_model,  if_auc  = train_isolation_forest(X_train, y_train, X_test, y_test)
    ae_model,  ae_auc  = train_autoencoder(X_train, y_train, X_test, y_test)

    build_ieee_graph()

    ensemble_scores, ens_auc = evaluate_ensemble(
        xgb_model, if_model, ae_model, X_test, y_test
    )

    registry = {
        'dataset':       'IEEE-CIS Fraud Detection',
        'trained_at':    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'n_features':    len(feature_names),
        'train_samples': int(len(X_train)),
        'test_samples':  int(len(X_test)),
        'fraud_rate':    float(y_train.mean()),
        'models': {
            'xgboost':          {'auc': round(xgb_auc, 4), 'file': 'xgboost_ieee.pkl'},
            'isolation_forest': {'auc': round(if_auc,  4), 'file': 'isolation_forest_ieee.pkl'},
            'autoencoder':      {'auc': round(ae_auc,  4) if ae_auc else None, 'file': 'autoencoder_ieee.keras'},
            'ensemble':         {'auc': round(ens_auc, 4)},
        }
    }

    with open(MODEL_DIR / 'registry_ieee.json', 'w') as f:
        json.dump(registry, f, indent=2)

    log.info('')
    log.info('=' * 60)
    log.info('ALL MODELS TRAINED SUCCESSFULLY!')
    log.info('=' * 60)
    log.info(f'  XGBoost AUC-ROC:          {xgb_auc:.4f}')
    log.info(f'  Isolation Forest AUC-ROC: {if_auc:.4f}')
    if ae_auc:
        log.info(f'  Autoencoder AUC-ROC:      {ae_auc:.4f}')
    else:
        log.info(f'  Autoencoder:              Skipped (TensorFlow not installed)')
    log.info(f'  Ensemble AUC-ROC:         {ens_auc:.4f}')
    log.info('')
    log.info(f'  Models saved to: models/ieee/')
    log.info(f'  Registry:        models/ieee/registry_ieee.json')
