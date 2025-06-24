# *******IMPORTANT*******
# this file do prediction on an a *single file* , not on folder
# for files in folder emotion prediction other code is provided in other python file 

# make sure to run the code in python notebook(colab, kaggle, jupyter etc) format as it will give error if tried to run via terminal
# make sure all the files are in same folder 
# this file supports VS code python notebook environment type file location (if user wants to run it in default way) else change all paths accordingly.
# update MODEL_PATH , SCALER_PATH , ENCODER_PATH , MAX_LEN_PATH according to environment being used 
# for example , if ran on kaggle the syntax for audio_path becomes , "/kaggle/working/"audio_name"
# suggested : run it on VS code for easy path definition




import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Step 1: Load Model and Preprocessing Assets ---
MODEL_PATH = "final_optuna_cnn_lstm.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
MAX_LEN_PATH = "max_len.npy"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
MAX_LEN = int(np.load(MAX_LEN_PATH))

# --- Step 2: Feature Extraction Configuration ---
SR = 16000
N_MFCC = 40
USE_DELTAS = True
USE_CHROMA = True
USE_CONTRAST = True

# --- Step 3: Optional RAVDESS emotion mapping ---
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
}

# --- Step 4: Feature Extraction ---
def extract_features(y_audio, sr, max_len):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    features = [mfcc]

    if USE_DELTAS:
        features.append(librosa.feature.delta(mfcc))
        features.append(librosa.feature.delta(mfcc, order=2))

    if USE_CHROMA or USE_CONTRAST:
        stft = np.abs(librosa.stft(y_audio))

    if USE_CHROMA:
        features.append(librosa.feature.chroma_stft(S=stft, sr=sr))

    if USE_CONTRAST:
        features.append(librosa.feature.spectral_contrast(S=stft, sr=sr))

    stacked = np.vstack(features).T
    padded = pad_sequences([stacked], maxlen=max_len, padding='post', truncating='post', dtype='float32')[0]
    return padded

# --- Step 5: Predict Emotion for One File ---
def predict_single_file(file_path):
    try:
        if not os.path.isfile(file_path) or not file_path.lower().endswith(".wav"):
            print(" Invalid audio file path.")
            return

        print(f" Processing file: {file_path}")
        y_audio, sr = librosa.load(file_path, sr=SR)
        features = extract_features(y_audio, sr, MAX_LEN)
        features_scaled = scaler.transform(features).reshape(1, features.shape[0], features.shape[1])

        prediction = model.predict(features_scaled, verbose=0)
        pred_idx = np.argmax(prediction)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        print(f" Predicted Emotion: {pred_label}")

        return pred_label

    except Exception as e:
        print(f" Error during prediction: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    #  Replace this with your .wav file path
    audio_file = "03-02-01-01-01-01-01.wav"  # <<< Change this accordingly

    predict_single_file(audio_file)




# User gives → .wav file path
# ↓
# predict_single_file()
# ↓
# librosa.load + feature extraction
# ↓
# Model prediction
# ↓
# Print predicted emotion label
