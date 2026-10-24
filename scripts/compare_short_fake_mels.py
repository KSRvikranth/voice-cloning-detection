import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


AUDIO_FILES = [
    (
        r"dataset/external/synthetic/voicemaker/179015481949335548250tu0sjb-voicemaker.in-speech.mp3",
        "MISSED FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",
        "FAKE 1"
    ),

    (
        r"dataset/external/synthetic/voicemaker/179031391799619747fhp78u4s-voicemaker.in-speech.mp3",
        "FAKE 2"
    ),

    (
        r"dataset/external/synthetic/voicemaker/1790313937859540443gxwgthkd-voicemaker.in-speech.mp3",
        "FAKE 3"
    )
]


def create_mel(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Same padding/truncation used by the CNN
    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]


    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Same normalization used by CNN
    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-6
    )

    return mel_db


for audio_path, title in AUDIO_FILES:

    print("\nProcessing:", title)

    try:

        mel = create_mel(audio_path)

        print("Mel shape:", mel.shape)
        print("Mean:", np.mean(mel))
        print("Std:", np.std(mel))
        print("Minimum:", np.min(mel))
        print("Maximum:", np.max(mel))

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            mel,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel"
        )

        plt.colorbar(
            format="%+2.0f"
        )

        plt.title(title)

        plt.tight_layout()

        plt.show()

    except Exception as e:

        print("ERROR:", e)