
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# CONFIGURATION
# -----------------------------

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_CSV = Path("results/train.csv")
VAL_CSV = Path("results/validation.csv")
TEST_CSV = Path("results/test.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "voice_cnn.pth"


# -----------------------------
# DATASET
# -----------------------------

class AudioDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.label_map = {
            "real": 0,
            "fake": 1
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        file_path = Path(row["file"])
        label_text = str(row["label"]).lower()

        # Handle paths stored relative to project root
        if not file_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {file_path}"
            )

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Ensure fixed length
        if len(audio) < SAMPLES:
            audio = librosa.util.fix_length(
                audio,
                size=SAMPLES
            )
        else:
            audio = audio[:SAMPLES]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Normalize
        mel_db = (
            mel_db - mel_db.mean()
        ) / (
            mel_db.std() + 1e-6
        )

        # Shape: [1, mel bands, time]
        mel_tensor = torch.tensor(
            mel_db,
            dtype=torch.float32
        ).unsqueeze(0)

        label = self.label_map[label_text]

        return mel_tensor, torch.tensor(label)


# -----------------------------
# CNN MODEL
# -----------------------------

class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -----------------------------
# TRAINING
# -----------------------------

def run_epoch(model, loader, criterion, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for audio, labels in loader:

        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(audio)

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


def main():

    print("Using device:", DEVICE)

    print("Loading datasets...")

    train_dataset = AudioDataset(TRAIN_CSV)
    val_dataset = AudioDataset(VAL_CSV)
    test_dataset = AudioDataset(TEST_CSV)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = VoiceCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    print("Starting CNN training...")

    for epoch in range(EPOCHS):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Saved best model.")

    print()
    print("Training complete.")
    print("Best model saved to:", MODEL_PATH)

    # Test using the best saved model
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for audio, labels in test_loader:

            audio = audio.to(DEVICE)

            outputs = model(audio)

            predicted = outputs.argmax(dim=1)

            predictions.extend(
                predicted.cpu().numpy()
            )

            actual.extend(
                labels.numpy()
            )

    print()
    print("TEST RESULTS")
    print("=" * 50)

    print(
        "Test Accuracy:",
        accuracy_score(actual, predictions)
    )

    print(
        classification_report(
            actual,
            predictions,
            labels=[0, 1],
            target_names=["real", "fake"],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()