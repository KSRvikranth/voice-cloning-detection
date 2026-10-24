import librosa
import numpy as np
import torch
import torch.nn as nn



AUDIO_FILE = r"C:\Users\sagas\OneDrive\Desktop\Voice Cloning Detection\dataset\external\synthetic\voicemaker\179015481949335548250tu0sjb-voicemaker.in-speech.mp3"
MODEL_FILE = "models/voice_cnn_multisource.pth"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


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


def make_mel(audio):

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

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    mel_db = (mel_db - mean) / (std + 1e-6)

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def predict(model, audio):

    # Normal 5-second preprocessing
    padded_audio = audio.copy()

    if len(padded_audio) < SAMPLES:
        padded_audio = np.pad(
            padded_audio,
            (0, SAMPLES - len(padded_audio))
        )
    else:
        padded_audio = padded_audio[:SAMPLES]

    tensor = make_mel(padded_audio)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    real = probabilities[0].item()
    fake = probabilities[1].item()

    return real, fake


def main():

    device = torch.device("cpu")

    print("=" * 60)
    print("SHORT AUDIO DIAGNOSTIC")
    print("=" * 60)

    print("\nLoading model...")

    model = VoiceCNN()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    print("\nLoading audio...")

    audio, sr = librosa.load(
        AUDIO_FILE,
        sr=SAMPLE_RATE,
        mono=True
    )

    print("Sample rate:", sr)
    print(
        "Duration:",
        round(len(audio) / sr, 3),
        "seconds"
    )

    print("\nRunning prediction...")

    real, fake = predict(
        model,
        audio
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("Prediction:",
          "fake" if fake > real else "real")

    print(
        "Real probability:",
        round(real, 4)
    )

    print(
        "Fake probability:",
        round(fake, 4)
    )

    print("=" * 60)


if __name__ == "__main__":
    main()