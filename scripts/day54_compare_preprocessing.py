
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "results" / "day46_external_inventory.csv"
MODEL_PATH = ROOT / "models" / "voice_cnn_multisource.pth"
OUTPUT = ROOT / "results" / "day54_preprocessing_comparison.csv"

SR = 16000
SAMPLES = SR * 5
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


def resolve_path(value):
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    matches = list(
        (ROOT / "dataset" / "external").rglob(path.name)
    )
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Cannot uniquely find: {value}")


def prepare_tensor(audio, trim_silence):
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) == 0:
        raise ValueError("Audio is empty after trimming.")

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (
        mel_db.std() + 1e-6
    )

    return torch.tensor(
        mel_db, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)


def predict(model, path, trim_silence):
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    x = prepare_tensor(audio, trim_silence).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    return probs[1].item()


def main():
    df = pd.read_csv(INVENTORY)

    model = VoiceCNN().to(device)
    state = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    rows = []

    for _, row in df.iterrows():
        try:
            path = resolve_path(row["file"])
            actual = str(row["label"]).strip().lower()

            if actual not in ("real", "fake"):
                continue

            original_prob = predict(model, path, False)
            trimmed_prob = predict(model, path, True)

            rows.append({
                "file": path.name,
                "actual_label": actual,
                "original_fake_probability": original_prob,
                "original_prediction": (
                    "fake" if original_prob >= 0.5 else "real"
                ),
                "trimmed_fake_probability": trimmed_prob,
                "trimmed_prediction": (
                    "fake" if trimmed_prob >= 0.5 else "real"
                )
            })

            print(
                f"{path.name}: "
                f"original={original_prob:.4f}, "
                f"trimmed={trimmed_prob:.4f}"
            )

        except Exception as error:
            print(f"FAILED: {row['file']}: {error}")

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No files were successfully evaluated.")

    results.to_csv(OUTPUT, index=False)

    for name in ("original", "trimmed"):
        predictions = results[f"{name}_prediction"]
        labels = results["actual_label"]

        print(f"\n{name.upper()} PREPROCESSING")
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
        print(
            "Fake recall: "
            f"{recall_score(labels, predictions, pos_label='fake', zero_division=0):.4f}"
        )
        print(
            "Fake F1: "
            f"{f1_score(labels, predictions, pos_label='fake', zero_division=0):.4f}"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()