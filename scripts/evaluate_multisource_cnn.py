from pathlib import Path
import csv

import numpy as np
import librosa
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = (
    PROJECT_ROOT
    / "results"
    / "multisource_test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "voice_cnn_multisource.pth"
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MULTISOURCE CNN EVALUATION")
print("=" * 70)

print("\nDevice:", DEVICE)

if not MODEL_PATH.exists():

    print("\nERROR:")
    print("Model not found:")
    print(MODEL_PATH)

    print(
        "\nRun the training script first."
    )

    exit()


model = VoiceCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model = model.to(DEVICE)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# LOAD TEST CSV
# ============================================================

if not TEST_CSV.exists():

    print("\nERROR:")
    print("Test CSV not found:")
    print(TEST_CSV)

    exit()


rows = []

with open(
    TEST_CSV,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)


print(
    "\nTest files:",
    len(rows)
)


# ============================================================
# PREDICTION
# ============================================================

y_true = []
y_pred = []
y_probability = []

failed_files = 0


print("\nStarting evaluation...")

for index, row in enumerate(rows):

    file_path = (
        PROJECT_ROOT
        / row["file"]
    )

    true_label = (
        1
        if row["label"] == "fake"
        else 0
    )

    try:

        audio, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Fixed 5 seconds
        # ----------------------------------------------------

        if len(audio) < SAMPLES:

            audio = np.pad(
                audio,
                (0, SAMPLES - len(audio))
            )

        else:

            audio = audio[:SAMPLES]

        # ----------------------------------------------------
        # Mel spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        mean = mel_db.mean()
        std = mel_db.std()

        mel_db = (
            mel_db - mean
        ) / (std + 1e-6)

        tensor = torch.tensor(
            mel_db,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            fake_probability = (
                probabilities[0][1].item()
            )

        y_true.append(true_label)

        y_pred.append(prediction)

        y_probability.append(
            fake_probability
        )

    except Exception as e:

        failed_files += 1

        print(
            "\nFailed:",
            file_path
        )

        print(
            "Error:",
            e
        )

    # Progress
    if (index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(rows)}"
        )


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    "\nSuccessfully evaluated:",
    len(y_true)
)

print(
    "Failed files:",
    failed_files
)


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

confusion = confusion_matrix(
    y_true,
    y_pred
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


print(
    "\nAccuracy:",
    round(accuracy, 6)
)

print(
    "Precision:",
    round(precision, 6)
)

print(
    "Recall:",
    round(recall, 6)
)

print(
    "F1 Score:",
    round(f1, 6)
)

print(
    "ROC-AUC:",
    round(roc_auc, 6)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    "                Predicted"
)

print(
    "              Real   Fake"
)

print(
    f"Actual Real   "
    f"{confusion[0][0]:5d} "
    f"{confusion[0][1]:5d}"
)

print(
    f"Actual Fake   "
    f"{confusion[1][0]:5d} "
    f"{confusion[1][1]:5d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Real",
            "Fake"
        ],
        zero_division=0
    )
)


# ============================================================
# PREDICTION COUNTS
# ============================================================

real_predictions = sum(
    1 for x in y_pred
    if x == 0
)

fake_predictions = sum(
    1 for x in y_pred
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


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)