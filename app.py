import streamlit as st
import numpy as np
import librosa
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option ...',
        ('Emotion Recognition', 'View Source Code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('Try it yourself by uploading an audio file.')
        application()
    elif selected_box == 'View Source Code':
        st.code(get_file_content_as_string('app.py'))

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://github.com/mohamedminhaj/speech-emotion-detection-usingML/tree/main/' + path  # Replace with actual URL
    response = urllib.request.urlopen(url)
    return response.read().decode('utf-8')

@st.cache(show_spinner=False)
def load_emotion_model():
    model = load_model('my_model.h5')  # Ensure 'my_model.h5' is in the same directory or provide a valid path
    return model

def preprocess_audio_file(file):
    y, sr = librosa.load(file, sr=16000)
    y = librosa.util.normalize(y)

    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    features = np.hstack([mfcc, chroma, spectral_contrast, zero_crossing_rate, rms])
    return features

def application():
    model_load_state = st.text("Loading model...")
    model = load_emotion_model()
    model_load_state.text("Model loaded successfully!")

    file_to_be_uploaded = st.file_uploader("Choose an audio file...", type=['wav'])
    if file_to_be_uploaded is not None:
        st.audio(file_to_be_uploaded, format='audio/wav')
        features = preprocess_audio_file(file_to_be_uploaded)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        prediction = model.predict(features)
        predicted_emotion = np.argmax(prediction) + 1
        emotions = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fearful', 7: 'Disgust', 8: 'Surprised'}

        st.write(f'Predicted Emotion: **{emotions[predicted_emotion]}**')

if __name__ == "__main__":
    main()
