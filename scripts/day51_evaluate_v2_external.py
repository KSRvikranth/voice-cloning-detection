from pathlib import Path
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INVENTORY_CSV = ROOT / "results" / "day46_external_inventory.csv"
MODEL_PATH = ROOT / "models" / "voice_cnn_multisource_v2.pth"
OUTPUT_CSV = ROOT / "results" / "day51_v2_external_predictions.csv"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. SAME PREPROCESSING USED BY V2
# --------------------------------------------------

def preprocess_audio(file_path):
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove leading and trailing silence.
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Prevent an empty signal from reaching the model.
    if len(audio) == 0:
        raise ValueError("Audio is empty after silence trimming.")

    # Make every audio sample 5 seconds long.
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    # Convert audio into a Mel spectrogram.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize the spectrogram.
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Shape: [1, 1, 64, time]
    features = torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return features


# --------------------------------------------------
# 3. SAME CNN ARCHITECTURE
# --------------------------------------------------

class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------
# 4. LOAD INVENTORY AND MODEL
# --------------------------------------------------

def find_column(data, possible_names):
    for name in possible_names:
        if name in data.columns:
            return name

    raise ValueError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns: {list(data.columns)}"
    )



def resolve_file_path(value):
    path_text = str(value).strip()
    path = Path(path_text)

    # 1. Check the supplied path directly.
    if path.is_file():
        return path

    # 2. Check a project-relative path.
    path = ROOT / path_text
    if path.is_file():
        return path

    # 3. Search the external audio folder by filename.
    external_folder = ROOT / "dataset" / "external"

    matches = list(external_folder.rglob(Path(path_text).name))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple files found with this name: {path_text}"
        )

    raise FileNotFoundError(
        f"Audio file not found: {path_text}\n"
        f"Searched under: {external_folder}"
    )

def convert_label(value):
    label = str(value).strip().lower()

    if label in ("real", "0"):
        return 0

    if label in ("fake", "1", "synthetic"):
        return 1

    raise ValueError(f"Unknown label: {value}")


print("=" * 65)
print("DAY 51 - V2 EXTERNAL EVALUATION")
print("=" * 65)
print("Device:", DEVICE)

data = pd.read_csv(INVENTORY_CSV)

file_column = find_column(data, ["file", "file_path", "path", "audio_path"])
label_column = find_column(data, ["label", "actual_label", "ground_truth"])

print("External files:", len(data))
print("Loading V2 model...")

model = VoiceCNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Support either a plain state dictionary or a wrapped checkpoint.
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 5. PREDICT EACH EXTERNAL RECORDING
# --------------------------------------------------

external_results = []
failed_files = 0
start_time = time.time()

for index, row in data.iterrows():
    try:
        file_path = resolve_file_path(row[file_column])
        true_label = convert_label(row[label_column])

        features = preprocess_audio(file_path).to(DEVICE)

        with torch.no_grad():
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)[0]

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        predicted_label = (
            1 if fake_probability >= 0.5 else 0
        )

        external_results.append({
            "file": str(file_path),
            "actual_label": "fake" if true_label == 1 else "real",
            "predicted_label": (
                "fake" if predicted_label == 1 else "real"
            ),
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "correct": bool(true_label == predicted_label)
        })

    except Exception as error:
        failed_files += 1
        print(f"\nFailed: {row[file_column]}")
        print("Reason:", error)

    if (index + 1) % 2 == 0 or index + 1 == len(data):
        print(f"Processed {index + 1}/{len(data)}")


# --------------------------------------------------
# 6. SAVE AND DISPLAY RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(external_results)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 65)
print("DAY 51 RESULTS")
print("=" * 65)

print("Successfully evaluated:", len(results_df))
print("Failed files:", failed_files)
print(f"Time taken: {time.time() - start_time:.1f} seconds")

if len(results_df) > 0:
    actual = results_df["actual_label"].map(
        {"real": 0, "fake": 1}
    ).to_numpy()

    predicted = results_df["predicted_label"].map(
        {"real": 0, "fake": 1}
    ).to_numpy()

    fake_probabilities = results_df["fake_probability"].to_numpy()

    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    print(
        "Fake precision:",
        f"{precision_score(actual, predicted, zero_division=0):.4f}"
    )
    print(
        "Fake recall:",
        f"{recall_score(actual, predicted, zero_division=0):.4f}"
    )
    print(
        "Fake F1:",
        f"{f1_score(actual, predicted, zero_division=0):.4f}"
    )

    if len(np.unique(actual)) == 2:
        print(
            "ROC-AUC:",
            f"{roc_auc_score(actual, fake_probabilities):.4f}"
        )

    print("\nConfusion matrix [real, fake]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0
        )
    )

    print("\nPer-file predictions:")
    print(results_df.to_string(index=False))

print("\nSaved results to:")
print(OUTPUT_CSV)
print("\nDAY 51 COMPLETE")