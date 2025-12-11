
# NeuroLearn — Streamlit App (Simple DNN)

This is a lightweight, demo-ready project that implements an AI-powered personalized learning platform prototype for neurodiverse students.
It uses a simple Keras model for student-content recommendations and a Streamlit UI (teacher & student views).

## Contents
- `app.py` - Main Streamlit app (run: `streamlit run app.py`)
- `train.py` - Script to train a very small Keras model on synthetic data
- `synthetic_data.py` - Utilities to create a synthetic interactions dataset
- `model_utils.py` - Helper functions to save/load model and embeddings
- `requirements.txt` - Python dependencies
- `report.pdf` - Short project report (or `report.md` if PDF generation unavailable)
- `README.md` - This file

## Quick start (recommended)
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. (Optional) Train the simple model:
   ```bash
   python train.py --epochs 5
   ```
   This will create `models/simple_model.h5`.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- The model in this repository is intentionally simple so it can run on a laptop without GPU.
- The Streamlit UI provides a teacher dashboard and a student view, plus controls to generate synthetic data and to run a training run.
- For a production system, replace the synthetic data with real, consented datasets and add privacy/federation features.
