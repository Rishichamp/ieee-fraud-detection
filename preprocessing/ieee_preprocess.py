"""
IEEE-CIS Fraud Detection — Preprocessing Pipeline
===================================================
Reads the 4 raw CSV files, merges identity + transaction tables,
engineers key features, handles missing values, encodes categoricals,
and produces train/test splits ready for all 4 models.

Dataset facts:
  - 590,540 transactions  (train: 590,540 | test: 506,691 — no labels)
  - 20,663 fraud cases (3.5%) — much richer than creditcard.csv (0.17%)
  - 434 raw features: transaction (394) + identity (41)
  - 4 files: train_transaction, train_identity, test_transaction, test_identity

Run:
    python preprocessing/ieee_preprocess.py

Output:
    data/ieee_cis/processed/
        X_train.npy, y_train.npy
        X_test.npy,  y_test.npy
        feature_names.json
        scaler.pkl
        label_encoders.pkl
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('ieee_preprocess')

DATA_DIR = Path(os.getenv('DATA_DIR', 'data/ieee_cis'))
OUT_DIR  = DATA_DIR / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAW FILES
# ─────────────────────────────────────────────────────────────────────────────
def load_raw():
    log.info('Loading raw CSV files...')

    train_txn = pd.read_csv(DATA_DIR / 'train_transaction.csv')
    train_id  = pd.read_csv(DATA_DIR / 'train_identity.csv')

    log.info(f'  train_transaction: {train_txn.shape}')
    log.info(f'  train_identity:    {train_id.shape}')
    log.info(f'  Fraud rate: {train_txn["isFraud"].mean()*100:.2f}%')

    # Merge on TransactionID (left join — not all transactions have identity)
    train = train_txn.merge(train_id, on='TransactionID', how='left')
    log.info(f'  Merged shape: {train.shape}')

    return train

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────
#
# IEEE-CIS has rich, named feature groups unlike creditcard.csv's PCA columns:
#
# TransactionDT  — timedelta from reference datetime (not actual timestamp)
# TransactionAmt — transaction amount in USD
# ProductCD      — product code (W, H, C, S, R)
# card1-card6    — card info: number, CVV, expiry, type, bank, etc.
# addr1, addr2   — billing address fields
# dist1, dist2   — distances (cardholder to merchant, etc.)
# P_emaildomain  — purchaser email domain (gmail.com, yahoo.com, etc.)
# R_emaildomain  — recipient email domain
# C1-C14         — counts (number of addresses linked to card, etc.)
# D1-D15         — timedeltas (days between events)
# M1-M9          — match features (name, address, etc.) — boolean-like
# V1-V339        — Vesta-engineered features (anonymous)
# id_01-id_38    — identity features (device, browser, OS, screen, etc.)
# DeviceType     — desktop / mobile
# DeviceInfo     — raw device string

CAT_COLS = [
    'ProductCD',
    'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
    'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo',
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info('Engineering features...')

    # ── Time features ─────────────────────────────────────────────────────────
    # TransactionDT is seconds since reference. Extract hour/day-of-week.
    df['hour']       = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    log.info('  ✔ Time features: hour, day_of_week, is_night')

    # ── Amount features ───────────────────────────────────────────────────────
    df['log_amount'] = np.log1p(df['TransactionAmt'])
    df['amount_cents'] = (df['TransactionAmt'] % 1 * 100).round(0)
    df['is_round_amount'] = (df['TransactionAmt'] % 1 == 0).astype(int)
    log.info('  ✔ Amount features: log_amount, amount_cents, is_round_amount')

    # ── Card aggregation features ─────────────────────────────────────────────
    # card1 = card number hash — use to compute per-card statistics
    card_stats = df.groupby('card1')['TransactionAmt'].agg(
        card_mean_amt='mean',
        card_std_amt='std',
        card_count='count'
    ).reset_index()
    df = df.merge(card_stats, on='card1', how='left')
    df['amt_vs_card_mean'] = df['TransactionAmt'] / (df['card_mean_amt'] + 1)
    log.info('  ✔ Card stats: card_mean_amt, card_std_amt, card_count, amt_vs_card_mean')

    # ── D-column normalisation ────────────────────────────────────────────────
    # D1 = days since card was first used. Subtracting from TransactionDT
    # makes it more stable across the dataset timeline.
    for i in range(1, 16):
        col = f'D{i}'
        if col in df.columns:
            df[f'{col}_norm'] = df['TransactionDT'] / (3600*24) - df[col]
    log.info('  ✔ D-column normalisation: D1_norm through D15_norm')

    # ── Email domain features ─────────────────────────────────────────────────
    # High-risk domains vs trusted domains
    trusted_domains = {'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com'}
    df['p_email_trusted'] = df['P_emaildomain'].isin(trusted_domains).astype(int)
    df['r_email_trusted'] = df['R_emaildomain'].isin(trusted_domains).astype(int)
    df['email_domain_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
    log.info('  ✔ Email features: p_email_trusted, r_email_trusted, email_domain_match')

    # ── Device features ───────────────────────────────────────────────────────
    df['is_mobile'] = (df['DeviceType'] == 'mobile').astype(int)
    log.info('  ✔ Device features: is_mobile')

    # ── Transaction count per addr ────────────────────────────────────────────
    addr_counts = df.groupby('addr1')['TransactionID'].count().reset_index()
    addr_counts.columns = ['addr1', 'addr1_txn_count']
    df = df.merge(addr_counts, on='addr1', how='left')
    log.info('  ✔ Address features: addr1_txn_count')

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log.info('Handling missing values...')

    # Drop columns with >90% missing — not informative
    thresh = len(df) * 0.10
    before = df.shape[1]
    df = df.dropna(axis=1, thresh=thresh)
    after = df.shape[1]
    log.info(f'  Dropped {before - after} columns with >90% missing')

    # Categorical: fill with 'unknown'
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    # Numerical: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    log.info(f'  Missing values remaining: {df.isnull().sum().sum()}')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple:
    log.info('Encoding categorical features...')
    encoders = {}

    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    log.info(f'  Encoded {len(encoders)} categorical columns')
    return df, encoders

# ─────────────────────────────────────────────────────────────────────────────
# 6. SELECT FEATURES & SCALE
# ─────────────────────────────────────────────────────────────────────────────
def prepare_final(df: pd.DataFrame) -> tuple:
    log.info('Preparing final feature matrix...')

    # Drop non-feature columns
    drop_cols = ['TransactionID', 'isFraud', 'TransactionDT']
    y = df['isFraud'].values if 'isFraud' in df.columns else None
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    log.info(f'  Final feature count: {len(feature_names)}')

    # Scale: StandardScaler on all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, scaler

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(X_train, X_test, y_train, y_test, feature_names, scaler, encoders):
    log.info('Saving processed data...')

    np.save(OUT_DIR / 'X_train.npy', X_train)
    np.save(OUT_DIR / 'X_test.npy',  X_test)
    np.save(OUT_DIR / 'y_train.npy', y_train)
    np.save(OUT_DIR / 'y_test.npy',  y_test)

    with open(OUT_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    joblib.dump(scaler,   OUT_DIR / 'scaler.pkl')
    joblib.dump(encoders, OUT_DIR / 'label_encoders.pkl')

    log.info(f'  X_train: {X_train.shape}')
    log.info(f'  X_test:  {X_test.shape}')
    log.info(f'  Fraud rate (train): {y_train.mean()*100:.2f}%')
    log.info(f'  Fraud rate (test):  {y_test.mean()*100:.2f}%')
    log.info(f'  Saved to {OUT_DIR}')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info('═' * 60)
    log.info('IEEE-CIS Fraud Detection — Preprocessing Pipeline')
    log.info('═' * 60)

    df = load_raw()
    df = engineer_features(df)
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)
    X, y, feature_names, scaler = prepare_final(df)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    save_outputs(X_train, X_test, y_train, y_test, feature_names, scaler, encoders)

    log.info('')
    log.info('✅ Preprocessing complete!')
    log.info(f'   Features: {len(feature_names)}')
    log.info(f'   Train:    {X_train.shape[0]:,} samples')
    log.info(f'   Test:     {X_test.shape[0]:,} samples')
    log.info('')
    log.info('Next step:')
    log.info('   python scripts/train_ieee.py')
