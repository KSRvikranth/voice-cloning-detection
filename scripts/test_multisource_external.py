import os
import numpy as np
import librosa
import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = "models/voice_cnn_multisource.pth"
AUDIO_FILES = [
    {
        "path": r"dataset/external/synthetic/voicemaker/1790060397128368606xgj5lw6p-voicemaker.in-speech.mp3",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/179015481949335548250tu0sjb-voicemaker.in-speech.mp3",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/179031391799619747fhp78u4s-voicemaker.in-speech.mp3",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/1790313937859540443gxwgthkd-voicemaker.in-speech.mp3",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/You’re the blueprint—don’t let...m4a",
        "actual": "fake"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/WhatsApp Ptt 2026-09-22 at 12.46.27 PM.ogg",
        "actual": "real"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/WhatsApp-Audio-2026-09-22-at-10.13.48-AM.wav",
        "actual": "real"
    },
    {
        "path": r"dataset/external/synthetic/voicemaker/WhatsApp Audio 2026-09-22 at 10.13.48 AM.mpeg",
        "actual": "real"
    }
]

MODEL_FILE = "models/voice_cnn_multisource.pth"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


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


# ============================================================
# PREPROCESS AUDIO
# ============================================================

def preprocess_audio(audio_file):

    print("\nLoading audio...")

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    original_samples = len(audio)

    print("Sample rate:", sr)
    print("Original samples:", original_samples)
    print(
        "Original duration:",
        round(original_samples / sr, 3),
        "seconds"
    )

    # --------------------------------------------------------
    # Pad or truncate to 5 seconds
    # --------------------------------------------------------

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    mel_db = (
        mel_db - mean
    ) / (std + 1e-6)

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# TEST ONE AUDIO
# ============================================================

def test_audio(model, audio_info):
    audio_path = audio_info["path"]
    actual = audio_info["actual"]

    print("\n" + "=" * 70)
    print("FILE:", audio_path)

    try:
        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        duration = len(audio) / SAMPLE_RATE

        print(f"Duration: {duration:.2f} seconds")
        print(f"Actual: {actual}")

        input_tensor = preprocess_audio(audio_path)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

        real_probability = probabilities[0][0].item()
        fake_probability = probabilities[0][1].item()

        prediction = "fake" if fake_probability > real_probability else "real"

        correct = prediction == actual

        print(f"Prediction: {prediction}")
        print(f"Real probability: {real_probability:.4f}")
        print(f"Fake probability: {fake_probability:.4f}")
        print(f"Correct: {'YES' if correct else 'NO'}")

        return {
            "file": audio_path,
            "actual": actual,
            "prediction": prediction,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "duration": duration,
            "correct": correct
        }

    except Exception as e:
        print("ERROR:", e)

        return {
            "file": audio_path,
            "actual": actual,
            "prediction": "ERROR",
            "real_probability": 0,
            "fake_probability": 0,
            "duration": 0,
            "correct": False
        }
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTISOURCE EXTERNAL AUDIO EVALUATION")
    print("=" * 70)

    device = torch.device("cpu")

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
    print(f"Testing {len(AUDIO_FILES)} external files...")

    results = []

    for audio_info in AUDIO_FILES:
        result = test_audio(model, audio_info)
        results.append(result)

    print("\n")
    print("=" * 70)
    print("FINAL EXTERNAL TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    fake_results = [r for r in results if r["actual"] == "fake"]
    real_results = [r for r in results if r["actual"] == "real"]

    fake_correct = sum(1 for r in fake_results if r["correct"])
    real_correct = sum(1 for r in real_results if r["correct"])

    print(f"Total files: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")

    print(f"\nOverall accuracy: {correct / total * 100:.2f}%")

    if fake_results:
        print(
            f"Fake detection: "
            f"{fake_correct}/{len(fake_results)} "
            f"= {fake_correct / len(fake_results) * 100:.2f}%"
        )

    if real_results:
        print(
            f"Real detection: "
            f"{real_correct}/{len(real_results)} "
            f"= {real_correct / len(real_results) * 100:.2f}%"
        )

    print("\nDetailed results:")
    print("-" * 100)

    print(
        f"{'File':<45} "
        f"{'Actual':<8} "
        f"{'Pred':<8} "
        f"{'Fake Prob':<12} "
        f"{'Duration':<10} "
        f"{'Correct'}"
    )

    print("-" * 100)

    for r in results:

        filename = os.path.basename(r["file"])

        if len(filename) > 42:
            filename = filename[:39] + "..."

        print(
            f"{filename:<45} "
            f"{r['actual']:<8} "
            f"{r['prediction']:<8} "
            f"{r['fake_probability']:<12.4f} "
            f"{r['duration']:<10.2f} "
            f"{'YES' if r['correct'] else 'NO'}"
        )


if __name__ == "__main__":
    main()