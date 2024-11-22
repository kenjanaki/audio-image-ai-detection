import json
import jsonpickle
import xgboost as xgb
import joblib
import librosa
import numpy as np

# Convert JSON model to Pickle
def convert_json_to_pickle(json_file, pickle_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Pickle the JSON data
    python_object = jsonpickle.decode(json.dumps(json_data))
    pickle_data = jsonpickle.encode(python_object)
    
    with open(pickle_file, 'wb') as f:
        f.write(pickle_data.encode('utf-8'))
    print(f"JSON file converted to pickle successfully at {pickle_file}.")

# Save model as Joblib file
def save_model_as_joblib(json_file, joblib_file):
    model = xgb.XGBClassifier()
    model.load_model(json_file)
    joblib.dump(model, joblib_file)
    print(f"Model saved as joblib successfully at {joblib_file}.")

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

# Preprocess the audio file and extract features
def preprocess_and_extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    y, _ = librosa.effects.trim(y)  # Trim silence
    max_len = sr * 5  # 5 seconds padding/truncating
    y = np.pad(y, (0, max_len - len(y)), mode='constant') if len(y) < max_len else y[:max_len]
    
    # Extract features
    mfcc = extract_mfcc(y)
    delta_mfcc = extract_delta(y)
    mel_spectrogram = extract_mel_spectrogram(y)
    zcr = extract_zero_crossing_rate(y)
    chroma = extract_chroma(y)
    cqt = extract_cqt(y)

    # Combine all features into one vector
    return np.concatenate([mfcc, delta_mfcc, mel_spectrogram, zcr, chroma, cqt])

# Predict the class for an audio file using a pre-trained model
def predict_audio_class(file, model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = preprocess_and_extract_features(file)
    features = np.array([features])  # Convert to 2D array for prediction
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction, probability

# Example usage
audio_file = "path_to_audio_file"  # Replace with the path to your audio file
model_path = "path_to_model"  # Replace with the path to your XGBoost model file

predicted_class, predicted_proba = predict_audio_class(audio_file, model_path)

print(f"Predicted class: {predicted_class}")
print(f"Predicted probabilities: {predicted_proba}")
