import numpy as np
from preprocessing.preprocess import load_and_preprocess
from models.xgboost_model import train_xgboost
from models.isolation_forest import train_isolation_forest
from models.autoencoder import train_autoencoder
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = load_and_preprocess()

xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
if_model, if_scores = train_isolation_forest(X_train, X_test, y_test)
ae_model, ae_scores = train_autoencoder(X_train, y_train, X_test, y_test)

# Normalize all scores to [0,1]
scaler = MinMaxScaler()
xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
if_norm  = scaler.fit_transform(if_scores.reshape(-1,1)).ravel()
ae_norm  = scaler.fit_transform(ae_scores.reshape(-1,1)).ravel()

# Weighted ensemble
ensemble = 0.5 * xgb_scores + 0.25 * if_norm + 0.25 * ae_norm
auc = roc_auc_score(y_test, ensemble)
print(f"\nEnsemble AUC-ROC: {auc:.4f}")