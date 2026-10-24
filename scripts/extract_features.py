from pathlib import Path
import librosa
import numpy as np
import pandas as pd


INPUT_DIR = Path("dataset/preprocessed")
OUTPUT_FILE = Path("results/features.csv")

SAMPLE_RATE = 16000


def extract_features(file):
    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=40
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_mean = np.mean(mel_db, axis=1)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    spectral_centroid_mean = np.mean(
        spectral_centroid
    )

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    spectral_bandwidth_mean = np.mean(
        spectral_bandwidth
    )

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(
        audio
    )

    zero_crossing_mean = np.mean(
        zero_crossing
    )

    # RMS Energy
    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)

    features = {}

    # Store MFCC features
    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}"] = value

    # Store Mel features
    for i, value in enumerate(mel_mean):
        features[f"mel_{i+1}"] = value

    features["spectral_centroid"] = spectral_centroid_mean
    features["spectral_bandwidth"] = spectral_bandwidth_mean
    features["zero_crossing_rate"] = zero_crossing_mean
    features["rms_energy"] = rms_mean

    return features


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for category in ["real", "fake"]:

        folder = INPUT_DIR / category

        if not folder.exists():
            print("Folder not found:", folder)
            continue

        files = list(folder.glob("*.wav"))

        print()
        print(category.upper())
        print("Files:", len(files))

        for i, file in enumerate(files, start=1):

            try:

                features = extract_features(file)

                features["file"] = str(file)
                features["label"] = category

                rows.append(features)

            except Exception as e:

                print("Error:", file)
                print(e)

            if i % 500 == 0 or i == len(files):
                print(
                    f"Processed {i}/{len(files)}"
                )

    dataframe = pd.DataFrame(rows)

    # Put file and label at the beginning
    columns = ["file", "label"]

    other_columns = [
        column
        for column in dataframe.columns
        if column not in columns
    ]

    dataframe = dataframe[
        columns + other_columns
    ]

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 50)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 50)

    print("Total samples:", len(dataframe))
    print("Total features:", len(dataframe.columns) - 2)

    print()
    print("Saved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()