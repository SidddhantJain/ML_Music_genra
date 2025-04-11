import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define your dataset path here
DATA_DIR = "Data/genres_original"
OUTPUT_CSV = "genre_features.csv"

# Define feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Pad or trim to 30 seconds
        if len(y) < sr * 30:
            y = np.pad(y, (0, sr * 30 - len(y)))
        else:
            y = y[:sr * 30]

        features = {
            "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            "rms": np.mean(librosa.feature.rms(y=y)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        }

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f"mfcc{i}"] = np.mean(mfcc[i - 1])

        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Loop through dataset
rows = []
genres = os.listdir(DATA_DIR)

for genre in tqdm(genres, desc="Processing genres"):
    genre_dir = os.path.join(DATA_DIR, genre)
    if not os.path.isdir(genre_dir):
        continue
    for file in os.listdir(genre_dir):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(genre_dir, file)
        feats = extract_features(file_path)
        if feats is not None:
            feats['genre'] = genre
            rows.append(feats)

# Create and save DataFrame
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Feature extraction completed! Saved to {OUTPUT_CSV}")
