
import os
from pathlib import Path

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def save_model(model, name='simple_model.h5'):
    path = MODEL_DIR / name
    model.save(path)
    return str(path)

def load_model(name='simple_model.h5'):
    from tensorflow.keras.models import load_model
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}')
    return load_model(path)
