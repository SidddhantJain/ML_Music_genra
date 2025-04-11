import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Fun facts for genres
genre_facts = {
    "rock": "🎸 Rock music often revolves around the electric guitar, and it originated in the 1950s.",
    "pop": "🎤 Pop music is catchy and often features simple lyrics and strong rhythms.",
    "classical": "🎻 Classical music is known for its complex compositions and orchestration.",
    "hiphop": "🎧 Hip-hop emphasizes rhythm, rhyme, and street culture.",
    "jazz": "🎷 Jazz features improvisation and swing notes, originating in the early 20th century.",
    "metal": "🤘 Metal is loud, aggressive, and often features powerful vocals and distorted guitars.",
    "country": "🪕 Country music blends folk and blues, with themes of love, heartbreak, and rural life.",
    "blues": "🎼 Blues is known for its soulful and melancholic sound and originated in the Deep South.",
    "reggae": "🌴 Reggae is a Jamaican genre characterized by offbeat rhythms and laid-back vibes.",
    "disco": "🕺 Disco is dance music from the 1970s with four-on-the-floor beats and funky basslines."
}

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Feature extractor
def extract_features_from_audio(y, sr):
    if len(y) < sr * 5:
        return None, None  # too short
    y = y[:sr * 30] if len(y) > sr * 30 else np.pad(y, (0, sr * 30 - len(y)))
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
    return np.array(features).reshape(1, -1), features

# UI Setup
st.set_page_config(page_title="🎶 Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier App")
st.write("Upload one or more `.wav` files and choose a model to predict the music genre.")

uploaded_files = st.file_uploader("Upload audio files", type=["wav"], accept_multiple_files=True)

model_choice = st.sidebar.selectbox("Choose a model", list(model_files.keys()))
model_path = os.path.join(MODELS_DIR, model_files[model_choice])
with open(model_path, "rb") as f:
    model = pickle.load(f)

for uploaded_file in uploaded_files:
    st.header(f"🎼 {uploaded_file.name}")
    y, sr = librosa.load(uploaded_file, sr=None)

    if len(y) < sr * 5:
        st.warning("⚠️ Audio too short (must be at least 5 seconds). Skipping.")
        continue

    st.audio(uploaded_file, format="audio/wav")

    # Waveform
    st.subheader("Waveform")
    fig_wave, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig_wave)

    # Spectrogram
    st.subheader("Spectrogram")
    fig_spec, ax2 = plt.subplots()
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log", ax=ax2)
    fig_spec.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogram")
    st.pyplot(fig_spec)

    # Features
    features_scaled, features_raw = extract_features_from_audio(y, sr)
    if features_scaled is None:
        st.error("Could not extract features properly.")
        continue
    features_scaled = scaler.transform(features_scaled)

    # Radar chart
    st.subheader("🎛️ Feature Overview")
    radar_labels = ['Chroma', 'RMS', 'Centroid', 'Bandwidth', 'Roll-off', 'ZCR'] + [f"MFCC{i}" for i in range(1, 21)]
    radar_df = pd.DataFrame([features_raw], columns=radar_labels)
    sns.set(style="whitegrid")
    radar_plot = sns.lineplot(data=radar_df.T, legend=False)
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())
    plt.clf()

    if st.button(f"Predict Genre for {uploaded_file.name}"):
        pred = model.predict(features_scaled)
        genre = encoder.inverse_transform(pred)[0]

        st.balloons()
        st.success(f"🎧 Predicted Genre: **{genre.upper()}**")
        fact = genre_facts.get(genre.lower(), "🎶 Enjoy the rhythm!")
        st.info(f"**Did you know?** {fact}")

        # Confidence bar chart
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            conf_df = pd.DataFrame({
                "Genre": encoder.inverse_transform(np.arange(len(proba))),
                "Confidence": proba
            }).sort_values("Confidence", ascending=False)
            st.subheader("📊 Confidence Levels")
            st.bar_chart(conf_df.set_index("Genre"))

        # Save to session state
        st.session_state.history.append({
            "File": uploaded_file.name,
            "Model": model_choice,
            "Predicted Genre": genre,
        })

        # Download CSV
        csv_data = pd.DataFrame(
            [np.append(features_raw, genre)],
            columns=[f"Feature_{i}" for i in range(len(features_raw))] + ["Predicted Genre"]
        )
        st.download_button("⬇️ Download Features & Prediction", csv_data.to_csv(index=False),
                           file_name=f"{uploaded_file.name}_prediction.csv", mime="text/csv")

if st.session_state.history:
    st.subheader("📝 Past Predictions This Session")
    st.table(pd.DataFrame(st.session_state.history))

st.sidebar.info("👈 Upload files, select a model, and hit predict!")
