import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16

TEST_CSV = "results/multisource_test.csv"
MODEL_PATH = "models/voice_cnn_multisource_v2.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DAY 50 - EVALUATE MULTISOURCE CNN V2")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# V2 PREPROCESSING
# ============================================================

def preprocess_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Trim silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # Pad or truncate to exactly 5 seconds

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    # Mel spectrogram

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Convert to decibels

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize

    mean = mel_db.mean()
    std = mel_db.std()

    mel_db = (mel_db - mean) / (std + 1e-6)

    # Convert to tensor

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    # [1, 64, time]

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    
    def __getitem__(self, index):

        row = self.data.iloc[index]

        file_path = row["file"]

        label_text = str(row["label"]).strip().lower()

        if label_text == "real":
            label = 0
        elif label_text == "fake":
            label = 1
        else:
            raise ValueError(
                f"Unknown label '{row['label']}' at row {index}"
            )

        features = preprocess_audio(file_path)

        return features, label


# ============================================================
# CNN MODEL
# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading test dataset...")

dataset = AudioDataset(TEST_CSV)

print("Test files:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading V2 model...")

model = VoiceCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_fake_probabilities = []

failed_files = 0

print()
print("Starting evaluation...")
print()


with torch.no_grad():

    for batch_number, (features, labels) in enumerate(loader):

        try:

            features = features.to(device)

            outputs = model(features)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            fake_probabilities = probabilities[:, 1]

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_fake_probabilities.extend(
                fake_probabilities.cpu().numpy()
            )

        except Exception as e:

            failed_files += len(labels)

        if (batch_number + 1) % 20 == 0:

            print(
                "Processed:",
                min(
                    (batch_number + 1) * BATCH_SIZE,
                    len(dataset)
                ),
                "/",
                len(dataset)
            )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("DAY 50 V2 EVALUATION RESULTS")
print("=" * 70)

print()

print(
    "Successfully evaluated:",
    len(all_labels)
)

print(
    "Failed files:",
    failed_files
)


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=1,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_fake_probabilities
)


print()
print("Accuracy :", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Fake Recall:", round(recall, 6))
print("Fake F1  :", round(f1, 6))
print("ROC-AUC  :", round(roc_auc, 6))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("Confusion Matrix:")
print()

print("                Predicted")
print("              Real   Fake")

print(
    "Actual Real ",
    cm[0][0],
    " ",
    cm[0][1]
)

print(
    "Actual Fake ",
    cm[1][0],
    " ",
    cm[1][1]
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)


# ============================================================
# PREDICTION COUNTS
# ============================================================

real_predictions = sum(
    1 for x in all_predictions
    if x == 0
)

fake_predictions = sum(
    1 for x in all_predictions
    if x == 1
)

print(
    "Predicted Real:",
    real_predictions
)

print(
    "Predicted Fake:",
    fake_predictions
)


print()
print("=" * 70)
print("DAY 50 COMPLETE")
print("=" * 70)