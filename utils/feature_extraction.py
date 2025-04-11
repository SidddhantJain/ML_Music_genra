import numpy as np
import librosa

def extract_features(file):
    y, sr = librosa.load(file, duration=30)
    if len(y) < sr * 30:
        y = np.pad(y, (0, max(0, sr * 30 - len(y))))
    else:
        y = y[:sr * 30]

    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    features.extend(mfccs_mean)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    features.extend(chroma_mean)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    features.extend(contrast_mean)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    features.extend(tonnetz_mean)

    return np.array(features)
