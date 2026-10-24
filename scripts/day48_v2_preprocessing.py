import librosa
import numpy as np
import torch


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

DURATION = 5

SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

TRIM_TOP_DB = 30


# ============================================================
# V2 PREPROCESSING
# ============================================================

def preprocess_audio_v2(audio_path):

    # --------------------------------------------------------
    # 1. LOAD AUDIO
    # --------------------------------------------------------

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    # --------------------------------------------------------
    # 2. TRIM SILENCE
    # --------------------------------------------------------

    audio, _ = librosa.effects.trim(
        audio,
        top_db=TRIM_TOP_DB
    )


    # --------------------------------------------------------
    # 3. PAD OR TRUNCATE
    # --------------------------------------------------------

    if len(audio) < SAMPLES:

        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    else:

        audio = audio[:SAMPLES]


    # --------------------------------------------------------
    # 4. MEL SPECTROGRAM
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    # --------------------------------------------------------
    # 5. CONVERT TO DECIBELS
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # --------------------------------------------------------
    # 6. NORMALIZATION
    # --------------------------------------------------------

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-6
    )


    # --------------------------------------------------------
    # 7. CONVERT TO PYTORCH TENSOR
    # --------------------------------------------------------

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )


    # Add channel dimension

    tensor = tensor.unsqueeze(0)


    # Add batch dimension

    tensor = tensor.unsqueeze(0)


    return tensor


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_file = (
        "dataset/external/synthetic/voicemaker/"
        "179015481949335548250tu0sjb-voicemaker.in-speech.mp3"
    )

    print()
    print("=" * 70)
    print("DAY 48 - V2 PREPROCESSING TEST")
    print("=" * 70)

    print()

    print("Test file:")
    print(test_file)

    print()

    tensor = preprocess_audio_v2(
        test_file
    )

    print(
        "Tensor shape:",
        tensor.shape
    )

    print(
        "Tensor dtype:",
        tensor.dtype
    )

    print(
        "Tensor mean:",
        tensor.mean().item()
    )

    print(
        "Tensor standard deviation:",
        tensor.std().item()
    )

    print()

    print("V2 preprocessing completed successfully.")

    print()
    print("DAY 48 PREPROCESSING TEST COMPLETE")