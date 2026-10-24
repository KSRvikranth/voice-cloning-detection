
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INVENTORY = ROOT / "results" / "day46_external_inventory.csv"
MODEL_PATH = ROOT / "models" / "voice_cnn_multisource.pth"
OUTPUT_CSV = ROOT / "results" / "day52_v1_trimmed_external_predictions.csv"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
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
        return self.classifier(self.features(x))


def resolve_file_path(value):
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    matches = list((ROOT / "dataset" / "external").rglob(path.name))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely locate audio file: {value}"
    )


def load_audio_tensor(file_path):
    audio, _ = librosa.load(
        str(file_path), sr=SAMPLE_RATE, mono=True
    )

    # Same V2 preprocessing: trim silence, then pad or truncate.
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) == 0:
        raise ValueError("Audio is empty after trimming.")

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (
        mel_db.std() + 1e-6
    )

    return torch.tensor(
        mel_db, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)


def get_column(df, choices):
    for name in choices:
        if name in df.columns:
            return name
    raise ValueError(
        f"Missing required column. Available columns: {list(df.columns)}"
    )


def main():
    inventory = pd.read_csv(INVENTORY)

    file_col = get_column(
        inventory, ["file", "file_path", "path", "audio_path"]
    )
    label_col = get_column(
        inventory, ["label", "actual_label", "ground_truth"]
    )

    model = VoiceCNN().to(device)
    state = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    rows = []

    for _, row in inventory.iterrows():
        file_value = row[file_col]
        actual = str(row[label_col]).strip().lower()

        if actual not in ("real", "fake"):
            print(f"Skipping unknown label: {file_value}")
            continue

        try:
            file_path = resolve_file_path(file_value)
            x = load_audio_tensor(file_path).to(device)

            with torch.no_grad():
                probabilities = torch.softmax(
                    model(x), dim=1
                )[0]

            real_prob = probabilities[0].item()
            fake_prob = probabilities[1].item()
            predicted = "fake" if fake_prob >= 0.5 else "real"

            rows.append({
                "file": file_path.name,
                "actual_label": actual,
                "predicted_label": predicted,
                "real_probability": real_prob,
                "fake_probability": fake_prob
            })

            print(
                f"{file_path.name}: actual={actual}, "
                f"predicted={predicted}, fake_prob={fake_prob:.4f}"
            )

        except Exception as error:
            print(f"FAILED: {file_value}: {error}")

    if not rows:
        raise RuntimeError("No external audio files were evaluated.")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    y_true = results["actual_label"]
    y_pred = results["predicted_label"]

    print("\nV1 with trimmed preprocessing")
    print(f"Evaluated: {len(results)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Fake precision: "
        f"{precision_score(y_true, y_pred, pos_label='fake', zero_division=0):.4f}"
    )
    print(
        f"Fake recall: "
        f"{recall_score(y_true, y_pred, pos_label='fake', zero_division=0):.4f}"
    )
    print(
        f"Fake F1: "
        f"{f1_score(y_true, y_pred, pos_label='fake', zero_division=0):.4f}"
    )
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()