from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import numpy as np

def train_isolation_forest(X_train, X_test, y_test):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,   # ~fraud rate in dataset
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # Scores: more negative = more anomalous
    scores = -model.score_samples(X_test)
    auc = roc_auc_score(y_test, scores)
    print(f"Isolation Forest AUC-ROC: {auc:.4f}")

    return model, scores