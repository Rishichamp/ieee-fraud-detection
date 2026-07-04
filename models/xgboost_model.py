import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import joblib

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=577,   # handles class imbalance
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"XGBoost AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'models/xgboost.pkl')
    return model