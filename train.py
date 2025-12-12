
"""Train a tiny Keras model on synthetic interactions.
Creates models/simple_model.h5 after training.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
import synthetic_data as sd
from model_utils import save_model
import os

def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def featurize(df):
    # simple aggregator per session
    X = df[['time_spent','content_id']].copy()
    X['content_id'] = X['content_id'] / (df['content_id'].max() + 1)
    X = X.values.astype('float32')
    y = (df['score'] > 0.6).astype('float32').values
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    df = sd.generate_synthetic_interactions(num_students=50, sessions_per_student=25)
    X, y = featurize(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=64)
    os.makedirs('models', exist_ok=True)
    save_model(model)
    print('Model trained and saved to models/simple_model.h5')
