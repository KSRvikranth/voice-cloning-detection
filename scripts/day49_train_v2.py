import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_WORKERS = 0
SEED = 42

MODEL_PATH = "models/voice_cnn_multisource_v2.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("=" * 70)
print("DAY 49 - TRAINING MULTISOURCE CNN V2")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# V2 PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # TRIM SILENCE
    # --------------------------------------------------------

    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # --------------------------------------------------------
    # PAD OR TRUNCATE
    # --------------------------------------------------------

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-6
    )

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class VoiceDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        audio_path = row["file"]

        label_text = str(row["label"]).lower()

        if label_text == "real":
            label = 0

        else:
            label = 1

        tensor = preprocess_audio(
            audio_path
        )

        return tensor, torch.tensor(
            label,
            dtype=torch.long
        )


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = VoiceDataset(
    "results/multisource_train.csv"
)

validation_dataset = VoiceDataset(
    "results/multisource_validation.csv"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print()
print("Training files:", len(train_dataset))
print("Validation files:", len(validation_dataset))


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

            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
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

        x = self.pool(x)

        x = self.classifier(x)

        return x


model = VoiceCNN().to(device)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_validation_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_correct = 0
    validation_total = 0
    validation_loss = 0.0

    with torch.no_grad():

        for inputs, labels in validation_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)


    train_accuracy = (
        train_correct / train_total
    )

    validation_accuracy = (
        validation_correct / validation_total
    )


    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss / len(train_loader):.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss / len(validation_loader):.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "Best V2 model saved."
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("DAY 49 TRAINING COMPLETE")
print("=" * 70)

print()
print(
    "Best Validation Accuracy:",
    f"{best_validation_accuracy:.4f}"
)

print()
print("Model saved:")
print(MODEL_PATH)

print()
print("V1 model remains untouched:")
print(
    "models/voice_cnn_multisource.pth"
)

print()
print("DAY 49 COMPLETE")