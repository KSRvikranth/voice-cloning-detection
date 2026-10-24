import librosa
import numpy as np
import torch
import torch.nn as nn

# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = "models/voice_cnn_multisource.pth"

AUDIO_FILE = (
    r"dataset/external/synthetic/voicemaker/"
    r"179015481949335548250tu0sjb-voicemaker.in-speech.mp3"
)


# =========================
# MODEL
# =========================

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
        x = self.classifier(x)

        return x


# =========================
# LOAD MODEL
# =========================

device = torch.device("cpu")

model = VoiceCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# =========================
# PREPROCESSING
# =========================

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

    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-6
    )

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor


# =========================
# PREDICTION
# =========================

def predict(audio):

    # Make exactly 5 seconds
    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    tensor = make_mel(audio)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    real_probability = probabilities[0].item()
    fake_probability = probabilities[1].item()

    return real_probability, fake_probability


# =========================
# LOAD AUDIO
# =========================

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

print()
print("Original duration:", len(audio) / SAMPLE_RATE)
print("Sample rate:", sr)


# =========================================================
# TEST 1
# ORIGINAL
# =========================================================

real, fake = predict(audio)

print()
print("TEST 1 - ORIGINAL")
print("Real:", round(real, 4))
print("Fake:", round(fake, 4))


# =========================================================
# TEST 2
# TRIM SILENCE
# =========================================================

trimmed, indexes = librosa.effects.trim(
    audio,
    top_db=30
)

print()
print("Trimmed duration:", len(trimmed) / SAMPLE_RATE)

real, fake = predict(trimmed)

print()
print("TEST 2 - TRIMMED SILENCE")
print("Real:", round(real, 4))
print("Fake:", round(fake, 4))


# =========================================================
# TEST 3
# TRIM + ADD SILENCE EQUALLY
# =========================================================

trimmed, indexes = librosa.effects.trim(
    audio,
    top_db=30
)

if len(trimmed) < SAMPLES:

    remaining = SAMPLES - len(trimmed)

    left = remaining // 2
    right = remaining - left

    centered = np.pad(
        trimmed,
        (left, right)
    )

else:

    centered = trimmed[:SAMPLES]


real, fake = predict(centered)

print()
print("TEST 3 - TRIM + CENTER")
print("Real:", round(real, 4))
print("Fake:", round(fake, 4))


# =========================================================
# TEST 4
# TRIM + REPEAT
# =========================================================

trimmed, indexes = librosa.effects.trim(
    audio,
    top_db=30
)

repeated = np.tile(
    trimmed,
    int(np.ceil(SAMPLES / len(trimmed)))
)

repeated = repeated[:SAMPLES]

real, fake = predict(repeated)

print()
print("TEST 4 - TRIM + REPEAT")
print("Real:", round(real, 4))
print("Fake:", round(fake, 4))