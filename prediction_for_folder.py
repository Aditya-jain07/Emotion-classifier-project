# *******IMPORTANT*******
# this file do prediction on an *input folder* , not on single file 
# for single file emotion prediction other code is provided in other python file 

# make sure to run the code in python notebook(colab, kaggle, jupyter etc) format as it will give error if tried to run via terminal
# make sure all the files are in same folder 
# this file supports VS code python notebook environment type file location (if user wants to run it in default way) else change all paths accordingly.
# update folder_path , MODEL_PATH , SCALER_PATH , ENCODER_PATH , MAX_LEN_PATH according to environment being used 
# for example , if ran on kaggle the syntax for folder_path becomes , "/kaggle/working/"folder_name"
# suggested : run it on VS code for easy path definition




import os
import numpy as np
import librosa
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --- Folder containing .wav files for prediction ---
folder_path = "Actor_01"  


# --- Step 1: Load Trained Model and Preprocessing Assets ---
MODEL_PATH = "final_optuna_cnn_lstm.h5"   # Trained CNN-LSTM model
SCALER_PATH = "scaler.pkl"                # Scaler used to normalize features
ENCODER_PATH = "label_encoder.pkl"        # Label encoder to decode predicted class
MAX_LEN_PATH = "max_len.npy"              # Max sequence length used during training

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
MAX_LEN = int(np.load(MAX_LEN_PATH))      # Max length to pad/truncate all sequences


# --- Step 2: Feature Extraction Settings ---
SR = 16000          # Sampling rate for audio
N_MFCC = 40         # Number of MFCC features
USE_DELTAS = True   # Use delta and delta-delta of MFCC
USE_CHROMA = True   # Use chroma features
USE_CONTRAST = True # Use spectral contrast


# --- Step 3: Emotion mapping (based on RAVDESS file naming convention) ---
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
}


# --- Step 4: Extract features from a single audio clip ---
def extract_features(y_audio, sr, max_len):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    features = [mfcc]

    # Add delta and delta-delta if enabled
    if USE_DELTAS:
        features.append(librosa.feature.delta(mfcc))
        features.append(librosa.feature.delta(mfcc, order=2))

    # Compute STFT once for chroma/contrast
    if USE_CHROMA or USE_CONTRAST:
        stft = np.abs(librosa.stft(y_audio))

    if USE_CHROMA:
        features.append(librosa.feature.chroma_stft(S=stft, sr=sr))

    if USE_CONTRAST:
        features.append(librosa.feature.spectral_contrast(S=stft, sr=sr))

    # Stack and pad to fixed shape (max_len x features)
    stacked = np.vstack(features).T
    padded = pad_sequences([stacked], maxlen=max_len, padding='post', truncating='post', dtype='float32')[0]
    return padded


# --- Step 5: Predict emotion from a single audio file ---
def predict_emotion(audio_path):
    try:
        # Load and preprocess audio
        y_audio, sr = librosa.load(audio_path, sr=SR)
        features = extract_features(y_audio, sr, MAX_LEN)

        # Apply saved scaler
        features_scaled = scaler.transform(features).reshape(1, features.shape[0], features.shape[1])

        # Predict and decode label
        prediction = model.predict(features_scaled, verbose=0)
        pred_idx = np.argmax(prediction)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        return pred_label
    except Exception as e:
        return f"ERROR: {e}"


# --- Step 6: Extract true label from filename (RAVDESS format) ---
def extract_true_label(filename):
    try:
        code = filename.split("-")[2]  # Emotion code is third item in RAVDESS filename
        return emotion_map.get(code)
    except:
        return None  # If filename doesn't follow RAVDESS format


# --- Step 7: Predict from all .wav files in a folder ---
def predict_from_folder(folder_path):
    y_true, y_pred = [], []

    print(f"\n🎧 Predicting emotions in: {folder_path}\n")
    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(".wav"):
            continue  # Skip non-wav files

        full_path = os.path.join(folder_path, file)
        pred = predict_emotion(full_path)
        print(f"{file} → {pred}")  # Show prediction

        # Try to get true label for evaluation (if available in filename)
        true_label = extract_true_label(file)
        if true_label and not str(pred).startswith("ERROR"):
            y_true.append(true_label)
            y_pred.append(pred)

    # Step 8: Print evaluation metrics if labels exist
    if y_true and y_pred:
        print("\n Evaluation Metrics (for labeled files):")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("\n Prediction complete — no labels detected in filenames, so evaluation skipped.")

# for above function -> if folder is from ravdess dataset then it'll given classification report as well , else only emotion will be printed


# --- Entry point ---
if __name__ == "__main__":
    if os.path.isdir(folder_path):
        predict_from_folder(folder_path)
    else:
        print(" Error: Provided path is not a valid directory.")


# workflow 
# entry point -> predict_from_folder -> predict_emotion , extract_true_labels -> extract_features