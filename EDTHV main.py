import webbrowser
import librosa

import soundfile
import time

import os
import glob
import pickle

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score


def extract_feature(file_name, mfcc, chroma, mel):

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if chroma:

            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


emotions = {

    '01': 'neutral',

    '02': 'calm',

    '03': 'happy',

    '04': 'sad',

    '05': 'angry',

    '06': 'fearful',

    '07': 'disgust',

    '08': 'surprised'
}
# Emotions to observe

# Load the data and extract features for each sound
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\ravdess\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


file = "C:\\ravdess\\Actor_02\\03-01-03-02-01-02-02.wav"

feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

#Split the dataset

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

# Initialize the Multi Layer Perceptron Classifier

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model

model.fit(x_train, y_train)

#Predict for the test set y_pred=model.predict(x_test)

y_pre = model.predict([feature])

print(y_pre)

time.sleep(2)


if y_pre[0] == "calm":

    webbrowser.open('C:\\ravdess\\photos\\Calm.html')

elif y_pre[0] == "neutral":

    webbrowser.open('C:\\ravdess\\photos\\Neutral.html')

elif y_pre[0] == "happy":

    webbrowser.open('C:\\ravdess\\photos\\Happy.html')

elif y_pre[0] == "sad":

    webbrowser.open('C:\\ravdess\\photos\\Sad.html')

elif y_pre[0] == "angry":

    webbrowser.open('C:\\ravdess\\photos\\Angry.html')

elif y_pre[0] == "fearful":

    webbrowser.open('C:\\ravdess\\photos\\Fearful.html')

elif y_pre[0] == "disgust":

    webbrowser.open('C:\\ravdess\\photos\\Disgust.html')

else:
    webbrowser.open('C:\\ravdess\\photos\\Error.html')