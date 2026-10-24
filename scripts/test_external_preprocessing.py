import os
import librosa
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = "models/voice_cnn_multisource.pth"


# =========================================================
# EXTERNAL AUDIO FILES
# =========================================================

AUDIO_FILES = [

    # =========================
    # FAKE
    # =========================

    (
        r"dataset/external/synthetic/voicemaker/"
        r"1790060397128368606xgj5lw6p-voicemaker.in-speech.mp3",
        "FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"179015481949335548250tu0sjb-voicemaker.in-speech.mp3",
        "FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",
        "FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"179031391799619747fhp78u4s-voicemaker.in-speech.mp3",
        "FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"1790313937859540443gxwgthkd-voicemaker.in-speech.mp3",
        "FAKE"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"You’re the blueprint—don’t let...m4a",
        "FAKE"
    ),


    # =========================
    # REAL
    # =========================

    (
        r"dataset/external/synthetic/voicemaker/"
        r"WhatsApp Ptt 2026-09-22 at 12.46.27 PM.ogg",
        "REAL"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"WhatsApp-Audio-2026-09-22-at-10.13.48-AM.wav",
        "REAL"
    ),

    (
        r"dataset/external/synthetic/voicemaker/"
        r"WhatsApp Audio 2026-09-22 at 10.13.48 AM.mpeg",
        "REAL"
    )
]


# =========================================================
# CNN MODEL
# =========================================================

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


# =========================================================
# LOAD MODEL
# =========================================================

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


# =========================================================
# CREATE MEL
# =========================================================

def create_mel(audio):

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


# =========================================================
# PREPARE AUDIO
# =========================================================

def prepare_audio(audio):

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    return audio


# =========================================================
# PREDICT
# =========================================================

def predict(audio):

    audio = prepare_audio(audio)

    tensor = create_mel(audio)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    real_probability = probabilities[0].item()

    fake_probability = probabilities[1].item()

    if fake_probability >= real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    return (
        prediction,
        real_probability,
        fake_probability
    )


# =========================================================
# MAIN TEST
# =========================================================

print()
print("=" * 90)
print("DAY 43 - EXTERNAL PREPROCESSING COMPARISON")
print("=" * 90)


for audio_path, actual_label in AUDIO_FILES:

    print()
    print("-" * 90)

    filename = os.path.basename(audio_path)

    print("File:", filename)
    print("Actual:", actual_label)

    if not os.path.exists(audio_path):

        print("ERROR: File not found")

        continue

    try:

        # ==========================================
        # LOAD
        # ==========================================

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        original_duration = len(audio) / SAMPLE_RATE

        print(
            "Original duration:",
            round(original_duration, 3)
        )


        # ==========================================
        # TEST 1
        # ORIGINAL
        # ==========================================

        prediction, real, fake = predict(audio)

        correct = prediction == actual_label

        print()
        print("ORIGINAL")

        print(
            "Prediction:",
            prediction
        )

        print(
            "Real:",
            round(real, 4)
        )

        print(
            "Fake:",
            round(fake, 4)
        )

        print(
            "Correct:",
            correct
        )


        # ==========================================
        # TRIM SILENCE
        # ==========================================

        trimmed_audio, indexes = librosa.effects.trim(
            audio,
            top_db=30
        )

        trimmed_duration = (
            len(trimmed_audio) / SAMPLE_RATE
        )

        print()
        print(
            "Trimmed duration:",
            round(trimmed_duration, 3)
        )


        # ==========================================
        # TEST 2
        # TRIMMED
        # ==========================================

        prediction, real, fake = predict(
            trimmed_audio
        )

        correct = prediction == actual_label

        print()
        print("TRIMMED SILENCE")

        print(
            "Prediction:",
            prediction
        )

        print(
            "Real:",
            round(real, 4)
        )

        print(
            "Fake:",
            round(fake, 4)
        )

        print(
            "Correct:",
            correct
        )


    except Exception as e:

        print()
        print("ERROR:")
        print(e)


print()
print("=" * 90)
print("DAY 43 COMPLETE")
print("=" * 90)