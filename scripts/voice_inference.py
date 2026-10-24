
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn


SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = Path("models/voice_cnn_multisource.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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



def prepare_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Match the V1 preprocessing: trim leading/trailing silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # Make the audio exactly 5 seconds long
    if len(audio) < SAMPLES:
        audio = librosa.util.fix_length(
            audio,
            size=SAMPLES
        )
    else:
        audio = audio[:SAMPLES]

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

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-6
    )

    mel_tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return mel_tensor


def predict_voice(audio_path):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    model = VoiceCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    audio_tensor = prepare_audio(audio_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(audio_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    real_probability = probabilities[0].item()
    fake_probability = probabilities[1].item()

    prediction = (
    "fake"
    if fake_probability >= real_probability
    else "real"
)

    confidence = max(real_probability, fake_probability)

    if confidence < 0.65:
        confidence_status = "UNCERTAIN"
    elif confidence < 0.80:
        confidence_status = "MODERATE"
    else:
        confidence_status = "HIGH_MODEL_CONFIDENCE"

    return {
        "prediction": prediction,
        "real_probability": round(real_probability, 4),
        "fake_probability": round(fake_probability, 4),
        "confidence": round(confidence, 4),
        "confidence_status": confidence_status
    }


if __name__ == "__main__":

    audio_path = Path("dataset/test/sample.wav")

    if not audio_path.exists():
        print("Audio file not found:", audio_path)
    else:
        result = predict_voice(audio_path)

        print("\n===== VOICE CNN PREDICTION =====")
        print(result)