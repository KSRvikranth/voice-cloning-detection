
from pathlib import Path
import json
from risk_scoring import calculate_risk
from explainable_warning import create_explanation
import whisper

from voice_inference import predict_voice
from social_engineering_detector import analyze_text


AUDIO_FILE = Path("dataset/test/sample.wav")
OUTPUT_FILE = Path("results/full_analysis.json")

WHISPER_MODEL = "base"


def transcribe_audio(audio_path):

    model = whisper.load_model(WHISPER_MODEL)

    result = model.transcribe(
        str(audio_path),
        fp16=False
    )

    return result["text"].strip()


def main():

    if not AUDIO_FILE.exists():
        print("Audio file not found:", AUDIO_FILE)
        return

    print("Running CNN voice analysis...")
    voice_result = predict_voice(AUDIO_FILE)

    print("Transcribing audio with Whisper...")
    transcript = transcribe_audio(AUDIO_FILE)

    print("Analyzing transcript...")
    text_result = analyze_text(transcript)
    risk_result = calculate_risk(
    voice_result,
    text_result
    )
    explanation_result = create_explanation(
    risk_result
    )
    combined_result = {
        "audio_file": str(AUDIO_FILE),
        "voice_analysis": voice_result,
        "transcription": transcript,
        "text_analysis": text_result,
        "risk_analysis": risk_result,
        "explanation": explanation_result
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            combined_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n===== FULL ANALYSIS =====")

    print("\nVoice result:")
    print(voice_result)

    print("\nTranscription:")
    print(transcript)

    print("\nText status:")
    print(text_result["text_status"])

    print("\nRisk score:")
    print(risk_result["risk_score"])

    print("\nRisk level:")
    print(risk_result["risk_level"])

    print("\nReasons:")
    for reason in risk_result["reasons"]:
        print("-", reason)

        print("\nSuggested action:")
    print(risk_result["action"])

    # Day 21: Explainable warning
    print("\n===== EXPLAINABLE WARNING =====")

    print("\nSummary:")
    print(explanation_result["summary"])

    print("\nDetected signals:")

    if explanation_result["signal_explanations"]:
        for item in explanation_result["signal_explanations"]:
            print("-", item["signal"])
            print(" ", item["explanation"])
    else:
        print("- No configured warning signals detected.")
    print("\nDisclaimer:")
    print(explanation_result["disclaimer"])

    print("\nFull result saved to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()