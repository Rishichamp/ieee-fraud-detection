import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

def build_autoencoder(input_dim=30):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(128, activation='relu')(inp)
    x = keras.layers.Dense(64,  activation='relu')(x)
    x = keras.layers.Dense(8,   activation='relu')(x)   # bottleneck
    x = keras.layers.Dense(64,  activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    out = keras.layers.Dense(input_dim, activation='sigmoid')(x)

    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X_train, y_train, X_test, y_test):
    # Train ONLY on legitimate transactions
    X_legit = X_train[y_train == 0]

    model = build_autoencoder(input_dim=X_train.shape[1])
    model.fit(X_legit, X_legit,
              epochs=50, batch_size=256,
              validation_split=0.1, verbose=1)

    # Reconstruction error = anomaly score
    X_rec = model.predict(X_test)
    mse = np.mean(np.power(X_test - X_rec, 2), axis=1)
    auc = roc_auc_score(y_test, mse)
    print(f"Autoencoder AUC-ROC: {auc:.4f}")

    model.save('models/autoencoder.h5')
    return model, mse