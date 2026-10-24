import os
import librosa
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/voice_cnn_multisource.pth"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# EXTERNAL FILES WITH CORRECT GROUND TRUTH
# ============================================================

AUDIO_FILES = [
    (
        "dataset/external/synthetic/voicemaker/1790060397128368606xgj5lw6p-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/179015481949335548250tu0sjb-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/179031391799619747fhp78u4s-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/1790313937859540443gxwgthkd-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/WhatsApp Audio 2026-09-22 at 10.13.48 AM.mpeg",
        "real"
    ),

    (
        "dataset/external/synthetic/voicemaker/WhatsApp Ptt 2026-09-22 at 12.46.27 PM.ogg",
        "real"
    ),

    (
        "dataset/external/synthetic/voicemaker/WhatsApp-Audio-2026-09-22-at-10.13.48-AM.wav",
        "real"
    ),

    (
        "dataset/external/synthetic/voicemaker/You’re the blueprint—don’t let...m4a",
        "fake"
    )
]


# ============================================================
# CNN MODEL
# ============================================================

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
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VoiceCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")
print("Device:", device)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_audio(audio_path, trim_silence=False):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # ----------------------------------------
    # OPTIONAL SILENCE TRIMMING
    # ----------------------------------------

    if trim_silence:

        audio, _ = librosa.effects.trim(
            audio,
            top_db=30
        )

    # ----------------------------------------
    # PAD OR TRUNCATE TO 5 SECONDS
    # ----------------------------------------

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    # ----------------------------------------
    # MEL SPECTROGRAM
    # ----------------------------------------

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

    # ----------------------------------------
    # NORMALIZATION
    # ----------------------------------------

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
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

def predict(audio_path, trim_silence=False):

    tensor = preprocess_audio(
        audio_path,
        trim_silence
    )

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    real_probability = probabilities[0][0].item()
    fake_probability = probabilities[0][1].item()

    if fake_probability >= real_probability:

        prediction = "fake"

    else:

        prediction = "real"

    return prediction, real_probability, fake_probability


# ============================================================
# EVALUATION
# ============================================================

results = []

print()
print("=" * 80)
print("DAY 45 - CORRECTED EXTERNAL PREPROCESSING EVALUATION")
print("=" * 80)

print()
print("External audio files found:", len(AUDIO_FILES))


for index, (audio_path, actual_label) in enumerate(AUDIO_FILES, start=1):

    print()
    print("=" * 80)

    print(f"{index} / {len(AUDIO_FILES)}")

    filename = os.path.basename(audio_path)

    print("File:", filename)
    print("Actual:", actual_label.upper())

    if not os.path.exists(audio_path):

        print("ERROR: File not found")

        continue

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    original_prediction, original_real, original_fake = predict(
        audio_path,
        trim_silence=False
    )

    # --------------------------------------------------------
    # TRIMMED
    # --------------------------------------------------------

    trimmed_prediction, trimmed_real, trimmed_fake = predict(
        audio_path,
        trim_silence=True
    )

    print(
        f"Original: {original_prediction.upper()} "
        f"Fake = {original_fake:.4f}"
    )

    print(
        f"Trimmed : {trimmed_prediction.upper()} "
        f"Fake = {trimmed_fake:.4f}"
    )

    original_correct = (
        original_prediction == actual_label
    )

    trimmed_correct = (
        trimmed_prediction == actual_label
    )

    results.append({
        "file": filename,
        "actual": actual_label,

        "original_prediction": original_prediction,
        "original_fake_probability": original_fake,
        "original_correct": original_correct,

        "trimmed_prediction": trimmed_prediction,
        "trimmed_fake_probability": trimmed_fake,
        "trimmed_correct": trimmed_correct
    })


# ============================================================
# RESULTS
# ============================================================

df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

output_file = "results/day45_preprocessing_comparison.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# ORIGINAL METRICS
# ============================================================

original_accuracy = (
    df["original_correct"].mean()
)

trimmed_accuracy = (
    df["trimmed_correct"].mean()
)


fake_files = df[df["actual"] == "fake"]
real_files = df[df["actual"] == "real"]


original_fake_recall = (
    (
        fake_files["original_prediction"] == "fake"
    ).mean()
)


trimmed_fake_recall = (
    (
        fake_files["trimmed_prediction"] == "fake"
    ).mean()
)


original_real_recall = (
    (
        real_files["original_prediction"] == "real"
    ).mean()
)


trimmed_real_recall = (
    (
        real_files["trimmed_prediction"] == "real"
    ).mean()
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 80)
print("DAY 45 CORRECTED RESULTS")
print("=" * 80)

print()

print(
    f"Original Accuracy: "
    f"{original_accuracy:.4f}"
)

print(
    f"Trimmed Accuracy: "
    f"{trimmed_accuracy:.4f}"
)

print()

print(
    f"Original Fake Recall: "
    f"{original_fake_recall:.4f}"
)

print(
    f"Trimmed Fake Recall: "
    f"{trimmed_fake_recall:.4f}"
)

print()

print(
    f"Original Real Recall: "
    f"{original_real_recall:.4f}"
)

print(
    f"Trimmed Real Recall: "
    f"{trimmed_real_recall:.4f}"
)

print()

print("Saved:")
print(output_file)

print()
print("DAY 45 COMPLETE")