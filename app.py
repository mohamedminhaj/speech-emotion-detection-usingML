import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('my_model.h5')

# Define emotion labels (make sure they match your model's output labels)
emotions = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}

# Preprocess the audio file
def preprocess_audio(file):
    # Load audio file with librosa
    y, sr = librosa.load(file, sr=16000)

    # Add preprocessing steps like feature extraction (MFCC, chroma, etc.)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    # Combine all features into a single feature vector
    features = np.hstack([mfcc, chroma, spectral_contrast, zero_crossing_rate, rms])

    # Return the features reshaped for the model input
    return np.expand_dims(features, axis=0)

# Streamlit title
st.title('Audio Emotion Recognition')

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Preprocess the uploaded audio file
    features = preprocess_audio(uploaded_file)
    
    # Predict the emotion
    prediction = model.predict(features)
    
    # Show the predicted emotion
    predicted_emotion = emotions[np.argmax(prediction)]
    st.write(f"Predicted Emotion: {predicted_emotion}")
