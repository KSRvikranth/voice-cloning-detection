import os
import numpy as np
import librosa
import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "models/voice_cnn_multisource.pth"

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# MODEL
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
# PREPROCESSING
# ============================================================

def preprocess_audio(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    original_duration = len(audio) / sr

    # Pad or truncate to 5 seconds

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]

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

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor, original_duration


# ============================================================
# PREDICT ONE FILE
# ============================================================

def predict(model, audio_file):

    tensor, duration = preprocess_audio(audio_file)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    real_probability = probabilities[0].item()
    fake_probability = probabilities[1].item()

    if fake_probability > real_probability:

        prediction = "fake"

    else:

        prediction = "real"

    return (
        prediction,
        real_probability,
        fake_probability,
        duration
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXTERNAL VOICE EVALUATION")
    print("=" * 70)

    device = torch.device("cpu")

    model = VoiceCNN()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    print("\nMultisource CNN loaded successfully.")

    folders = [
        ("synthetic", "dataset/external/synthetic"),
        ("real", "dataset/external/real")
    ]

    total = 0
    correct = 0

    synthetic_total = 0
    synthetic_correct = 0

    real_total = 0
    real_correct = 0

    for expected_label, folder in folders:

        if not os.path.exists(folder):

            print("\nFolder not found:", folder)

            continue

        print("\n" + "-" * 70)
        print("EXPECTED:", expected_label.upper())
        print("FOLDER:", folder)
        print("-" * 70)

        for root, dirs, files in os.walk(folder):

            for filename in files:

                if not filename.lower().endswith(
                    (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
                ):
                    continue

                path = os.path.join(
                    root,
                    filename
                )

                try:

                    prediction, real_probability, fake_probability, duration = predict(
                        model,
                        path
                    )

                    total += 1

                    if prediction == expected_label:

                        correct += 1

                        if expected_label == "synthetic":

                            synthetic_correct += 1

                        else:

                            real_correct += 1

                    if expected_label == "synthetic":

                        synthetic_total += 1

                    else:

                        real_total += 1

                    print("\nFile:", filename)

                    print(
                        "Expected:",
                        expected_label
                    )

                    print(
                        "Prediction:",
                        prediction
                    )

                    print(
                        "Real:",
                        round(real_probability, 4)
                    )

                    print(
                        "Fake:",
                        round(fake_probability, 4)
                    )

                    print(
                        "Duration:",
                        round(duration, 2),
                        "seconds"
                    )

                except Exception as e:

                    print("\nERROR:", filename)

                    print(e)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nTotal files:", total)

    if total > 0:

        accuracy = correct / total

        print(
            "Overall accuracy:",
            round(accuracy * 100, 2),
            "%"
        )

    if synthetic_total > 0:

        synthetic_accuracy = (
            synthetic_correct /
            synthetic_total
        )

        print(
            "Synthetic detection accuracy:",
            round(synthetic_accuracy * 100, 2),
            "%"
        )

    if real_total > 0:

        real_accuracy = (
            real_correct /
            real_total
        )

        print(
            "Real detection accuracy:",
            round(real_accuracy * 100, 2),
            "%"
        )


if __name__ == "__main__":

    main()