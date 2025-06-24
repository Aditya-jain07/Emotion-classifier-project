import os
import numpy as np
import librosa
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load model and preprocessing assets ---
MODEL_PATH = "final_optuna_cnn_lstm.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
MAX_LEN_PATH = "max_len.npy"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
MAX_LEN = int(np.load(MAX_LEN_PATH))

# --- Feature extraction settings ---
SR = 16000
N_MFCC = 40
USE_DELTAS = True
USE_CHROMA = True
USE_CONTRAST = True

# --- Feature extraction function ---
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

# --- Prediction function ---
def predict_emotion(file):
    try:
        y_audio, sr = librosa.load(file, sr=SR)
        features = extract_features(y_audio, sr, MAX_LEN)
        features_scaled = scaler.transform(features).reshape(1, features.shape[0], features.shape[1])

        prediction = model.predict(features_scaled, verbose=0)
        pred_idx = np.argmax(prediction)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(np.max(prediction))

        return pred_label, confidence

    except Exception as e:
        return f"Error: {e}", None



# --- Streamlit UI ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title(" Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Analyzing..."):
        label, confidence = predict_emotion(uploaded_file)

    if confidence is not None:
        st.success(f" Predicted Emotion: **{label}**")
        st.write(f" Confidence: **{confidence * 100:.2f}%**")
    else:
        st.error(label)  # label holds error message here
