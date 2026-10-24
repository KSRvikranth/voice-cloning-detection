from pathlib import Path
import csv
import random

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "results" / "multisource_train.csv"
VAL_CSV = PROJECT_ROOT / "results" / "multisource_validation.csv"
TEST_CSV = PROJECT_ROOT / "results" / "multisource_test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "voice_cnn_multisource.pth"


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


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MULTISOURCE CNN TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, csv_path):

        self.records = []

        with open(
            csv_path,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                self.records.append(row)

        print(
            f"Loaded {len(self.records)} files from "
            f"{csv_path.name}"
        )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        row = self.records[index]

        file_path = PROJECT_ROOT / row["file"]

        label = 1 if row["label"] == "fake" else 0

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        try:

            audio, _ = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                mono=True
            )

        except Exception as e:

            print(
                "\nERROR loading:",
                file_path
            )

            print("Error:", e)

            # Return silent audio if one file fails
            audio = np.zeros(
                SAMPLES,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Make exactly 5 seconds
        # ----------------------------------------------------

        if len(audio) < SAMPLES:

            audio = np.pad(
                audio,
                (0, SAMPLES - len(audio))
            )

        else:

            audio = audio[:SAMPLES]

        # ----------------------------------------------------
        # Mel Spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # Convert to decibels
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

        # ----------------------------------------------------
        # Convert to Tensor
        # ----------------------------------------------------

        tensor = torch.tensor(
            mel_db,
            dtype=torch.float32
        )

        # Add channel dimension
        # [64, time] -> [1, 64, time]

        tensor = tensor.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return tensor, label


# ============================================================
# CNN MODEL
# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 3
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
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_dataset = AudioDataset(
    TRAIN_CSV
)

validation_dataset = AudioDataset(
    VAL_CSV
)

test_dataset = AudioDataset(
    TEST_CSV
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

model = VoiceCNN()

model = model.to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for batch_index, (audio, labels) in enumerate(
        train_loader
    ):

        audio = audio.to(DEVICE)

        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(audio)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # Progress
        if (
            batch_index + 1
        ) % 50 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )

    accuracy = correct / total

    average_loss = (
        total_loss /
        len(train_loader)
    )

    return average_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for audio, labels in loader:

            audio = audio.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(audio)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    average_loss = (
        total_loss /
        len(loader)
    )

    return average_loss, accuracy


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_validation_accuracy = 0.0


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 50)

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    validation_loss, validation_accuracy = (
        evaluate(validation_loader)
    )

    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "  
        f"{validation_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "\n*** Best model saved ***"
        )

        print(
            MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_accuracy = evaluate(
    test_loader
)

print(
    "\nTest Loss:",
    round(test_loss, 4)
)

print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)

print(
    "\nBest Validation Accuracy:",
    round(
        best_validation_accuracy,
        4
    )
)

print(
    "\nMultisource model saved to:"
)

print(MODEL_PATH)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)