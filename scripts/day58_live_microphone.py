
from pathlib import Path
import sys
import time

import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "voice_cnn_multisource.pth"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5
SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
THRESHOLD = 0.5

sys.path.insert(0, str(ROOT))


# This architecture must match the trained V1 model.

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

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def prepare_audio(audio):
    # Convert stereo microphone audio to mono.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Remove leading and trailing silence.
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Pad short audio or keep only the first five seconds.
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    # Create a Mel spectrogram using the training settings.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize the spectrogram.
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Shape: batch, channel, mel, time.
    tensor = torch.tensor(
        mel_db, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor


def main():
    if not MODEL_PATH.exists():
        print("Model file not found:", MODEL_PATH)
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = VoiceCNN().to(device)

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    print("V1 model loaded.")
    print("Starting local microphone detection.")
    print("Speak near your microphone.")
    print("Press Ctrl+C to stop.")
    print("This is a prototype, not a guarantee of authenticity.")

    try:
        while True:
            print("\nListening for 5 seconds...")

            start_time = time.time()

            audio = sd.rec(
                SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            audio = audio.flatten()
            tensor = prepare_audio(audio).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probabilities = F.softmax(logits, dim=1)[0]

            real_probability = probabilities[0].item()
            fake_probability = probabilities[1].item()

            if fake_probability >= THRESHOLD:
                prediction = "POTENTIALLY SYNTHETIC"
            else:
                prediction = "PREDICTED REAL"

            elapsed = time.time() - start_time

            print("Prediction:", prediction)
            print(f"Real probability: {real_probability:.4f}")
            print(f"Fake probability: {fake_probability:.4f}")
            print(f"Processing + capture time: {elapsed:.2f} seconds")
            print("Reminder: model prediction is not proof.")

    except KeyboardInterrupt:
        print("\nLive microphone detection stopped.")


if __name__ == "__main__":
    main()