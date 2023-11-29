
###################### Important Note:-  Run this file code only when want to train model for voice emotion detection################

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import librosa
import soundfile

# Function to extract features from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")

            n_fft = min(2048, len(X) // 4)
            n_fft = max(1, n_fft)

            sample_rate = sound_file.samplerate

            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

            if chroma:
                stft = np.abs(librosa.stft(X[:, 0], n_fft=n_fft))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X[:, 0], sr=sample_rate, n_mfcc=40, n_fft=n_fft).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X[:, 0], sr=sample_rate, n_fft=n_fft).T, axis=0)
                result = np.hstack((result, mel))
        return result
    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return None

# Dictionary mapping emotion codes to labels
emotions = {
    '01': 'neutral',
    '02': 'positive',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['neutral', 'positive', 'sad', 'angry']

# Function to load and preprocess data
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("/home/manish/Desktop/monteage/mont_project/speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        if feature is not None:
            x.append(feature)
            y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

if __name__ == "__main__":
    # Load and split the data
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)

    # Scale features to a common range
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create and train the model
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(500,), learning_rate='adaptive', max_iter=2000, tol=1e-4)
    model.fit(x_train_scaled, y_train)

    # Predict for the test set
    y_pred = model.predict(x_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=observed_emotions)

    # Print the results
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_mat)
