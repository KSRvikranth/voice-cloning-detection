
import whisper
import os
import sys

# Load Whisper model
# "base" is a reasonable starting point for a CPU laptop.
print("Loading Whisper model...")
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        print("Error: Audio file not found:", audio_path)
        return

    print("\nTranscribing:", audio_path)

    try:
        result = model.transcribe(
            audio_path,
            fp16=False
        )

        text = result["text"].strip()

        print("\n===== TRANSCRIPTION =====")
        print(text)

        # Save transcription
        os.makedirs("results/transcriptions", exist_ok=True)

        filename = os.path.splitext(
            os.path.basename(audio_path)
        )[0]

        output_path = (
            "results/transcriptions/"
            + filename
            + ".txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("\nSaved transcription to:", output_path)

    except Exception as e:
        print("Transcription failed:", e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "dataset/test/sample.wav"

    transcribe_audio(audio_file)