import os
import librosa
import numpy as np

SAMPLE_RATE = 16000

AUDIO_FILES = [
    r"dataset/external/synthetic/voicemaker/179015481949335548250tu0sjb-voicemaker.in-speech.mp3",

    r"dataset/external/synthetic/voicemaker/1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",

    r"dataset/external/synthetic/voicemaker/179031391799619747fhp78u4s-voicemaker.in-speech.mp3",

    r"dataset/external/synthetic/voicemaker/1790313937859540443gxwgthkd-voicemaker.in-speech.mp3"
]


def analyze_audio(path):

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(path))
    print("=" * 70)

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = len(audio) / sr

    rms = librosa.feature.rms(y=audio)[0]

    zero_crossing = librosa.feature.zero_crossing_rate(audio)[0]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]

    silence_threshold = 0.01

    silence_ratio = np.mean(
        np.abs(audio) < silence_threshold
    )

    print(f"Sample rate: {sr}")
    print(f"Samples: {len(audio)}")
    print(f"Duration: {duration:.3f} seconds")

    print(f"RMS mean: {np.mean(rms):.6f}")
    print(f"RMS max: {np.max(rms):.6f}")

    print(
        f"Zero crossing rate: "
        f"{np.mean(zero_crossing):.6f}"
    )

    print(
        f"Spectral centroid: "
        f"{np.mean(spectral_centroid):.2f} Hz"
    )

    print(
        f"Spectral bandwidth: "
        f"{np.mean(spectral_bandwidth):.2f} Hz"
    )

    print(
        f"Silence ratio: "
        f"{silence_ratio * 100:.2f}%"
    )


for file in AUDIO_FILES:

    try:
        analyze_audio(file)

    except Exception as e:

        print("ERROR:", e)