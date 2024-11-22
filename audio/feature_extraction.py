import os
import random
from glob import glob
import pandas as pd
import librosa
import numpy as np

# Define paths
librispeech_path = r"C:\path\to\LibriSpeech"
timit_path = r"C:\path\to\TIMIT-TTS"
save_dir = r"C:\path\to\save\features"

# Function to sample audio files
def get_sample_files(base_path, extension="*.flac", sample_size=5000):
    sampled_files = []
    for root, dirs, files in os.walk(base_path):
        audio_files = glob(os.path.join(root, extension))
        sampled_files.extend(random.sample(audio_files, min(len(audio_files), sample_size)))
    return sampled_files

# Get samples
real_audio_files = get_sample_files(librispeech_path, extension="*.flac", sample_size=5000)
fake_audio_files = get_sample_files(timit_path, extension="*.wav", sample_size=5000)

# Ensure balanced datasets
real_audio_files = random.sample(real_audio_files, 5000) if len(real_audio_files) > 5000 else real_audio_files
fake_audio_files = random.sample(fake_audio_files, 5000) if len(fake_audio_files) > 5000 else fake_audio_files

# Preprocess audio
def preprocess_audio(file_path, target_sr=16000, pad_length=5):
    y, sr = librosa.load(file_path, sr=target_sr)  # Resample to 16kHz
    y = librosa.util.normalize(y)  # Normalize to [-1, 1]
    y, _ = librosa.effects.trim(y)  # Trim silence
    max_len = target_sr * pad_length
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]  # Trim if longer
    return y, sr

# Feature extraction
def extract_features(y, sr):
    features = {}
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
    spec = librosa.stft(y)
    features['spectrogram'] = librosa.amplitude_to_db(np.abs(spec))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfccs'] = mfccs
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features['delta_mfccs'] = delta_mfccs
    features['delta2_mfccs'] = delta2_mfccs
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate'] = zcr
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_features'] = chroma
    cqt = np.abs(librosa.core.cqt(y, sr=sr))
    features['cqt'] = cqt
    log_spec = librosa.amplitude_to_db(np.abs(spec))
    features['log_spectrogram'] = log_spec
    return features

# Process files
def process_audio_files(audio_files):
    all_features = []
    for file in audio_files:
        print(f"Processing: {file}")
        y, sr = preprocess_audio(file)
        features = extract_features(y, sr)
        features['file_name'] = os.path.basename(file)
        all_features.append(features)
    return all_features

# Extract features for real and fake audio files
real_features = process_audio_files(real_audio_files)
fake_features = process_audio_files(fake_audio_files)

# Convert to DataFrames and save to CSV
real_df = pd.DataFrame(real_features)
fake_df = pd.DataFrame(fake_features)

real_df.to_csv(os.path.join(save_dir, "real_audio_features.csv"), index=False)
fake_df.to_csv(os.path.join(save_dir, "fake_audio_features.csv"), index=False)

print("Feature extraction and saving complete.")
