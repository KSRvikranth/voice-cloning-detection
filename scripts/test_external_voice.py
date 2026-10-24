import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.voice_inference import predict_voice


# ==========================================
# CHANGE THIS FILE NAME IF NEEDED
# ==========================================

audio_file = "dataset/voicemaker/1790060397128368606xgj5lw6p-voicemaker.in-speech.mp3"

# ==========================================
# USE MULTISOURCE MODEL
# ==========================================

model_path = "models/voice_cnn_multisource.pth"


print("=" * 60)
print("EXTERNAL AI VOICE TEST")
print("=" * 60)

print()
print("Audio file:", audio_file)
print("Model:", model_path)
print()

if not os.path.exists(audio_file):
    print("ERROR: Audio file not found.")
    print("Check the file path.")
    sys.exit()

if not os.path.exists(model_path):
    print("ERROR: Multisource model not found.")
    print("Check models/voice_cnn_multisource.pth")
    sys.exit()

print("Running prediction...")
print()

result = predict_voice(
    audio_file
)

print("=" * 60)
print("RESULT")
print("=" * 60)

print()

print("Prediction:", result["prediction"])
print("Real probability:", result["real_probability"])
print("Fake probability:", result["fake_probability"])

print()

if result["prediction"] == "FAKE":
    print("The multisource CNN classified this audio as AI-generated/synthetic.")
else:
    print("The multisource CNN classified this audio as real.")

print()
print("=" * 60)
print("IMPORTANT")
print("=" * 60)

print(
    "This is an external test sample and was not used during training."
)

print(
    "A single prediction does not prove that the detector can identify "
    "all AI-generated voices."
)