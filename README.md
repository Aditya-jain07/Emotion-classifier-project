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

