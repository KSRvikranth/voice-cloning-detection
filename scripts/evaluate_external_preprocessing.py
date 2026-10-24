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
# EXTERNAL DATASET
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
# CNN
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
# METRIC FUNCTION
# =========================================================

def calculate_metrics(results):

    total = len(results)

    correct = 0

    true_fake = 0
    false_fake = 0
    missed_fake = 0
    true_real = 0
    false_real = 0

    fake_probabilities = []

    for result in results:

        actual = result["actual"]
        prediction = result["prediction"]

        if actual == prediction:

            correct += 1

        if actual == "FAKE":

            fake_probabilities.append(
                result["fake_probability"]
            )

            if prediction == "FAKE":

                true_fake += 1

            else:

                missed_fake += 1

        else:

            if prediction == "REAL":

                true_real += 1

            else:

                false_real += 1


    # Fake precision
    if true_fake + false_real > 0:

        fake_precision = (
            true_fake /
            (true_fake + false_real)
        )

    else:

        fake_precision = 0


    # Fake recall
    if true_fake + missed_fake > 0:

        fake_recall = (
            true_fake /
            (true_fake + missed_fake)
        )

    else:

        fake_recall = 0


    # Fake F1
    if fake_precision + fake_recall > 0:

        fake_f1 = (
            2 *
            fake_precision *
            fake_recall /
            (fake_precision + fake_recall)
        )

    else:

        fake_f1 = 0


    # Real recall
    if true_real + false_real > 0:

        real_recall = (
            true_real /
            (true_real + false_real)
        )

    else:

        real_recall = 0


    accuracy = correct / total

    if fake_probabilities:

        average_fake_probability = (
            sum(fake_probabilities) /
            len(fake_probabilities)
        )

    else:

        average_fake_probability = 0


    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "true_fake": true_fake,
        "missed_fake": missed_fake,
        "true_real": true_real,
        "false_real": false_real,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "real_recall": real_recall,
        "average_fake_probability": average_fake_probability
    }


# =========================================================
# RUN EVALUATION
# =========================================================

original_results = []
trimmed_results = []


print()
print("=" * 90)
print("DAY 44 - EXTERNAL PREPROCESSING EVALUATION")
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

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        # =================================================
# ORIGINAL
# =================================================

        original_prediction, original_real, original_fake = predict(audio)

        original_results.append({
            "file": filename,
            "actual": actual_label,
            "prediction": original_prediction,
            "real_probability": original_real,
            "fake_probability": original_fake
        })


# =================================================
# TRIMMED
# =================================================

        trimmed_audio, indexes = librosa.effects.trim(
            audio,
            top_db=30
        )

        trimmed_prediction, trimmed_real, trimmed_fake = predict(
            trimmed_audio
        )

        trimmed_results.append({
            "file": filename,
            "actual": actual_label,
            "prediction": trimmed_prediction,
            "real_probability": trimmed_real,
            "fake_probability": trimmed_fake
        })


        print(  
            "Original:",
            original_prediction,
            "Fake =",
            round(original_fake, 4)
        )

        print(
            "Trimmed :",
            trimmed_prediction,
            "Fake =",
            round(trimmed_fake, 4)
        )
    except Exception as e:

        print("ERROR:", e)


# =========================================================
# CALCULATE METRICS
# =========================================================

original_metrics = calculate_metrics(
    original_results
)

trimmed_metrics = calculate_metrics(
    trimmed_results
)


# =========================================================
# DISPLAY ORIGINAL
# =========================================================

print()
print("=" * 90)
print("ORIGINAL PREPROCESSING RESULTS")
print("=" * 90)

print(
    "Total:",
    original_metrics["total"]
)

print(
    "Correct:",
    original_metrics["correct"]
)

print(
    "Incorrect:",
    original_metrics["incorrect"]
)

print(
    "Accuracy:",
    round(
        original_metrics["accuracy"],
        4
    )
)

print(
    "Fake Precision:",
    round(
        original_metrics["fake_precision"],
        4
    )
)

print(
    "Fake Recall:",
    round(
        original_metrics["fake_recall"],
        4
    )
)

print(
    "Fake F1:",
    round(
        original_metrics["fake_f1"],
        4
    )
)

print(
    "Real Recall:",
    round(
        original_metrics["real_recall"],
        4
    )
)

print(
    "Missed Fake:",
    original_metrics["missed_fake"]
)


# =========================================================
# DISPLAY TRIMMED
# =========================================================

print()
print("=" * 90)
print("TRIMMED PREPROCESSING RESULTS")
print("=" * 90)

print(
    "Total:",
    trimmed_metrics["total"]
)

print(
    "Correct:",
    trimmed_metrics["correct"]
)

print(
    "Incorrect:",
    trimmed_metrics["incorrect"]
)

print(
    "Accuracy:",
    round(
        trimmed_metrics["accuracy"],
        4
    )
)

print(
    "Fake Precision:",
    round(
        trimmed_metrics["fake_precision"],
        4
    )
)

print(
    "Fake Recall:",
    round(
        trimmed_metrics["fake_recall"],
        4
    )
)

print(
    "Fake F1:",
    round(
        trimmed_metrics["fake_f1"],
        4
    )
)

print(
    "Real Recall:",
    round(
        trimmed_metrics["real_recall"],
        4
    )
)

print(
    "Missed Fake:",
    trimmed_metrics["missed_fake"]
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print()
print("=" * 90)
print("FINAL COMPARISON")
print("=" * 90)

print(
    "Original Accuracy:",
    round(
        original_metrics["accuracy"],
        4
    )
)

print(
    "Trimmed Accuracy:",
    round(
        trimmed_metrics["accuracy"],
        4
    )
)

print()

print(
    "Original Fake Recall:",
    round(
        original_metrics["fake_recall"],
        4
    )
)

print(
    "Trimmed Fake Recall:",
    round(
        trimmed_metrics["fake_recall"],
        4
    )
)

print()

print(
    "Original Real Recall:",
    round(
        original_metrics["real_recall"],
        4
    )
)

print(
    "Trimmed Real Recall:",
    round(
        trimmed_metrics["real_recall"],
        4
    )
)

print()
print("=" * 90)
print("DAY 44 COMPLETE")
print("=" * 90)