import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Path of our test audio
file_path = "dataset/test/sample.wav"


# Load audio
audio, sr = librosa.load(
    file_path,
    sr=16000,
    mono=True
)


# Display basic information
print("Audio loaded successfully!")
print("Sampling Rate:", sr)
print("Number of Samples:", len(audio))
print("Duration:", len(audio) / sr, "seconds")


# Create Mel Spectrogram
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)


# Convert power spectrogram to decibels
mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)


# Display Mel Spectrogram
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()

plt.show()