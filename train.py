"""
Efficient Drug Discovery using Molecular Data - Training Script
Author: Dev Kapania | IIT Roorkee Research Intern
"""

import numpy as np
import os
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='DrugActivityDNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def train(X_train, X_val, y_train, y_val):
    model = build_model(X_train.shape[1])
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_auc', patience=15,
                                restore_best_weights=True, mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=7, min_lr=1e-6),
        callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.h5'),
                                  monitor='val_auc', save_best_only=True, mode='max')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=32,
        callbacks=cb, verbose=1
    )
    return model, history


def evaluate(model, X_test, y_test):
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print('\n' + '='*60)
    print('FINAL TEST SET EVALUATION')
    print('='*60)
    print(classification_report(y_test, y_pred, target_names=['Inactive', 'Active']))
    print(f'F1 Score : {f1_score(y_test, y_pred):.4f}')
    print(f'ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}')


if __name__ == '__main__':
    print('Loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print('Training DNN...')
    model, history = train(X_train, X_val, y_train, y_val)

    evaluate(model, X_test, y_test)

    model.save(os.path.join(MODELS_DIR, 'drug_dnn_final.h5'))
    print('\nModel saved! Training complete.')
