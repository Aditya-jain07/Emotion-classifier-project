# Emotion_classifier_prediction-project

### RAVDESS dataset used -> Audio_Song_Actors_01-24 + Audio_Speech_Actors_01-24


- Emotion_classifier_project.ipynb   -> main code file
- final_optuna_cnn_lstm.h5           -> final model
- scaler.pkl                         -> scaler used
- label_encoder.pkl                  -> label encoder used
- max_len.npy                        -> max number of time steps model will handle for each mfcc feature
- best_hyperparameters               -> best parameters saved after optuna tuning (not useful in code as final model is saved)
- prediction_for_folder.py           -> code to run predictions when input is provided in a way of "**`FOLDER`**"
- prediction_for_file.py             -> code to run predictions when input is in form of "**`SINGLE FILE`**" 
- app.py                             -> code for streamlit deployment of the model 
- 03-02-01-01-01-01-01.wav           -> test file 

- video link for visual representation of streamlit (only for IIT ROORKEE) -> https://drive.google.com/file/d/17-y_AlNXSkXOjK275aczR4ThDWoO2DOo/view?usp=drive_link


## 📌 Project Description

This project is a full pipeline for **Speech Emotion Recognition (SER)** using the **RAVDESS dataset**. It utilizes audio feature extraction (MFCCs, deltas, chroma, spectral contrast), deep learning models (CNN + Bidirectional LSTM), and hyperparameter optimization (Optuna). A Streamlit-based web app is also included for live inference.

---

## 🎯 Objective

To classify emotions in raw audio recordings into categories such as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## 🗃️ Dataset: RAVDESS

* Source: Ryerson Audio-Visual Database of Emotional Speech and Song
* Format: `.wav`
* File naming convention encodes emotion using a code (e.g., `03` = happy)

---

## 🔄 Preprocessing Pipeline

### 1. **Emotion Mapping**

Each `.wav` file's emotion is extracted using filename parsing and mapped using:

```python
emotion_map = {
  '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
  '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
}
```

### 2. **Label Encoding**

Emotions are encoded as integers using `LabelEncoder`. Saved for inference.

### 3. **Data Splitting**

* 70% training
* 15% validation
* 15% test
* Stratified to preserve emotion balance

### 4. **Audio Augmentation** (for training only)

* Pitch shifting
* Time stretching
* Gaussian noise

### 5. **Feature Extraction**

* **MFCCs** (Mel-frequency cepstral coefficients)
* **Deltas & Delta-deltas**
* **Chroma STFT**
* **Spectral Contrast**
* All features are stacked and padded to a fixed length (95th percentile)

### 6. **Standard Scaling**

* `StandardScaler` fitted on training set
* Applied to all sets (train, val, test)

---

## 🧠 Model Architecture

### CNN-BiLSTM Hybrid:

* **3 Convolutional Blocks**: Each with Conv1D, BatchNorm, MaxPooling, SpatialDropout
* **1 BiLSTM Layer**: Bidirectional LSTM with L2 regularization
* **Dense Layer**: Fully connected + Dropout
* **Output Layer**: Softmax for emotion classification

### Optuna for Hyperparameter Tuning

* Tuning of 15+ hyperparameters: conv filters, dropout, LSTM units, learning rate, regularization, etc.
* Objective: Maximize validation accuracy

---

## 🧪 Evaluation & Metrics

### ✅ Best Validation Accuracy (Optuna):

```
0.8261
```

### 📊 Final Test Accuracy:

```
~81% (exact value printed at runtime)
```

### 📝 Classification Report:

Includes precision, recall, F1-score for all 8 classes

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Angry     | 0.92      | 0.79   | 0.85     | 57      |
| Calm      | 0.87      | 0.84   | 0.85     | 55      |
| Disgust   | 0.79      | 0.90   | 0.84     | 29      |
| Fear      | 0.72      | 0.88   | 0.79     | 56      |
| Happy     | 0.78      | 0.63   | 0.70     | 57      |
| Neutral   | 0.80      | 0.71   | 0.75     | 28      |
| Sad       | 0.73      | 0.79   | 0.76     | 56      |
| Surprise  | 0.85      | 0.97   | 0.90     | 29      |
| **Overall Accuracy** |        |         | **0.80** | **367** |
| **Macro Avg**        | 0.81   | 0.81    | 0.81     | 367     |
| **Weighted Avg**     | 0.81   | 0.80    | 0.80     | 367     |


## 🧪 Best Hyperparameters (sample)

```json
{
  "conv1_filters": 224,
  "conv2_filters": 256,
  "conv3_filters": 128,
  "kernel_size": 7,
  "pool_size": 2,
  "conv_dropout": 0.2,
  "lstm_units": 224,
  "dense_units": 160,
  "dropout": 0.3,
  "learning_rate": 0.00046957,
  "batch_size": 16,
  "conv_l2": 2.9e-06,
  "dense_l2": 2.2e-06,
  "output_l2": 0.00186,
  "lstm_l2": 3.37e-05
}
```

---

## 🚀 How to Run the Project

### 1. use file provided named "**`prediction_for_folder`**" or "**`prediction_for_file`**"

### 2. Launch Streamlit App

* use file named "**`app.py`**" and run the below command in terminal
* usecase video link is provided above

```bash
streamlit run app.py
```

## 📁 Output Files

| File                        | Purpose                          |
| --------------------------- | -------------------------------- |
| `final_optuna_cnn_lstm.h5`  | Final trained model              |
| `scaler.pkl`                | StandardScaler for inference     |
| `label_encoder.pkl`         | LabelEncoder for class labels    |
| `max_len.npy`               | Max length for padding sequences |
| `best_hyperparameters.json` | Optuna best config               |

---

