import lightgbm as lgb
import numpy as np
import pandas as pd
import librosa
import os

# Feature extraction functions
def extract_mfcc(y):
    return np.mean(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).T, axis=0)

def extract_delta(y):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)).T, axis=0)

def extract_mel_spectrogram(y):
    return np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000), ref=np.max).T, axis=0)

def extract_zero_crossing_rate(y):
    return np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

def extract_chroma(y):
    return np.mean(librosa.feature.chroma_stft(y=y, sr=16000).T, axis=0)

def extract_cqt(y):
    return np.mean(np.abs(librosa.cqt(y, sr=16000)).T, axis=0)

# Preprocess the audio and extract features
def preprocess_and_extract_features(file):
    y, sr = librosa.load(file, sr=16000)  # Load the audio file

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad or truncate to 5 seconds
    max_len = sr * 5
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]

    # Extract features
    mfcc = extract_mfcc(y)
    delta_mfcc = extract_delta(y)
    mel_spectrogram = extract_mel_spectrogram(y)
    zcr = np.array([extract_zero_crossing_rate(y)])  # Make it a 1D array
    chroma = extract_chroma(y)
    cqt = extract_cqt(y)

    # Combine all extracted features into one feature vector (flattened)
    return np.concatenate([mfcc.flatten(), delta_mfcc.flatten(), mel_spectrogram.flatten(), zcr.flatten(), chroma.flatten(), cqt.flatten()])

# Load the trained LightGBM model
bst = lgb.Booster(model_file='lgb_audio_model.txt')

# Path to the audio files for prediction
audio_file = r"C:\path_to_audio_file\example_audio.wav"  # Update with your audio file path

# Feature extraction from the audio file
features = preprocess_and_extract_features(audio_file)

# Convert features to a 2D array (1 sample) for prediction
features = np.array([features])

# Make predictions using the trained LightGBM model
y_pred = bst.predict(features)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print the predicted class and probabilities
print(f"Predicted class: {y_pred_binary[0]}")
print(f"Predicted probabilities: {y_pred[0]}")
