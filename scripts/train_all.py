"""
Full training pipeline — run inside the trainer container.
Saves all model artifacts to MODEL_PATH (shared Docker volume).

Usage:
  docker compose --profile train up trainer
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("trainer")

DATA_PATH  = Path(os.getenv("DATA_PATH",  "data/creditcard.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    if not DATA_PATH.exists():
        log.error(f"Dataset not found at {DATA_PATH}")
        log.error("Download from: kaggle datasets download mlg-ulb/creditcardfraud")
        sys.exit(1)

    log.info(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    log.info(f"Shape: {df.shape}  Fraud rate: {df['Class'].mean()*100:.3f}%")

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")
    log.info("Saved scaler.pkl")

    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────────────────────────────────────
# 2. XGBoost
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    log.info("Training XGBoost...")

    # SMOTE on training set
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        log.info(f"  SMOTE: {len(y_res):,} samples (balanced)")
    except ImportError:
        log.warning("  imbalanced-learn not installed, skipping SMOTE")
        X_res, y_res = X_train, y_train

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum()),
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"  XGBoost AUC-ROC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, model.predict(X_test), digits=4))

    joblib.dump(model, MODEL_PATH / "xgboost.pkl")
    log.info("  Saved xgboost.pkl")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 3. Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
def train_isolation_forest(X_train, X_test, y_test):
    log.info("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    scores = -model.score_samples(X_test)
    auc = roc_auc_score(y_test, scores)
    log.info(f"  Isolation Forest AUC-ROC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH / "isolation_forest.pkl")
    log.info("  Saved isolation_forest.pkl")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 4. Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
def train_autoencoder(X_train, y_train, X_test, y_test):
    log.info("Training Autoencoder...")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        log.warning("  TensorFlow not installed, skipping autoencoder")
        return None

    X_legit = X_train[y_train == 0]
    inp_dim  = X_train.shape[1]

    inp = keras.Input(shape=(inp_dim,))
    x   = keras.layers.Dense(128, activation="relu")(inp)
    x   = keras.layers.Dropout(0.1)(x)
    x   = keras.layers.Dense(64,  activation="relu")(x)
    x   = keras.layers.Dense(8,   activation="relu")(x)
    x   = keras.layers.Dense(64,  activation="relu")(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(inp_dim, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    cb = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    model.fit(X_legit, X_legit,
              epochs=50, batch_size=256,
              validation_split=0.1,
              callbacks=cb, verbose=1)

    X_rec = model.predict(X_test, batch_size=512)
    mse   = np.mean(np.power(X_test - X_rec, 2), axis=1)
    auc   = roc_auc_score(y_test, mse)
    log.info(f"  Autoencoder AUC-ROC: {auc:.4f}")

    model.save(MODEL_PATH / "autoencoder.h5")
    log.info("  Saved autoencoder.h5")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save model registry (metadata)
# ─────────────────────────────────────────────────────────────────────────────
def save_registry(results: dict):
    import json, datetime
    registry = {
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "models": results
    }
    with open(MODEL_PATH / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"Saved registry.json → {registry}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    if_model  = train_isolation_forest(X_train, X_test, y_test)
    ae_model  = train_autoencoder(X_train, y_train, X_test, y_test)

    save_registry({
        "xgboost":          {"file": "xgboost.pkl"},
        "isolation_forest": {"file": "isolation_forest.pkl"},
        "autoencoder":      {"file": "autoencoder.h5"},
        "scaler":           {"file": "scaler.pkl"},
    })

    log.info("✅ All models trained and saved successfully.")
