import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_model.h5')

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    # Add your preprocessing steps
    return np.expand_dims(features, axis=0)

st.title('Audio Emotion Recognition')
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    features = preprocess_audio(uploaded_file)
    prediction = model.predict(features)
    st.write("Predicted Emotion:", np.argmax(prediction))