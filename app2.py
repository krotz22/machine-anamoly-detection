import streamlit as st
import numpy as np
import librosa
import pickle
import sounddevice as sd

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to preprocess and extract features
def preprocess_audio_for_rf(audio, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_mean, axis=0)

# Function to record audio from the microphone
def record_audio(duration=5, sr=22050):
    st.write("Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording completed.")
    return recording.flatten()

st.title('Valve Anomaly Detection with Random Forest')
st.write("Record audio or upload an audio file to classify it as normal or abnormal.")

# Buttons for uploading file or recording audio
option = st.selectbox("Choose an option", ["Upload File", "Record Audio"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    
    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=22050)
        features = preprocess_audio_for_rf(y)
        prediction = rf_model.predict(features)
        prediction_label = 'Abnormal' if prediction[0] == 1 else 'Normal'
        st.write(f"The predicted label is: {prediction_label}")

elif option == "Record Audio":
    duration = st.slider("Select recording duration (seconds)", 1, 10, 5)
    if st.button("Record"):
        y = record_audio(duration=duration)
        features = preprocess_audio_for_rf(y)
        prediction = rf_model.predict(features)
        prediction_label = 'Abnormal' if prediction[0] == 1 else 'Normal'
        st.write(f"The predicted label is: {prediction_label}")
