import streamlit as st
import librosa
import numpy as np
import pickle
import os

# Load models
MODELS_DIR = "models"
model_files = {
    "KNN": "KNN_model.pkl",
    "Random Forest": "RandomForest_model.pkl",
    "XGBoost": "XGBoost_model.pkl"
}

# Load scaler and label encoder
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

# Feature extractor
def extract_features_from_audio(y, sr):
    # Pad or trim
    if len(y) < sr * 30:
        y = np.pad(y, (0, sr * 30 - len(y)))
    else:
        y = y[:sr * 30]

    features = [
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y))
    ]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend([np.mean(mfcc[i]) for i in range(20)])

    return np.array(features).reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="🎶 Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier App")
st.write("Upload a `.wav` file and choose a model to predict the music genre.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format='audio/wav')

    # Feature extraction
    features = extract_features_from_audio(y, sr)
    features_scaled = scaler.transform(features)

    # Model selection
    model_choice = st.selectbox("Choose a model", list(model_files.keys()))
    model_path = os.path.join(MODELS_DIR, model_files[model_choice])
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prediction
    if st.button("Predict Genre"):
        pred = model.predict(features_scaled)
        genre = encoder.inverse_transform(pred)[0]
        st.success(f"🎧 Predicted Genre: **{genre.upper()}**")

