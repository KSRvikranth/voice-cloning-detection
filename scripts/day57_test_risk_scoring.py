
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.social_engineering_detector import analyze_text
from scripts.risk_scoring import calculate_risk


def test_case(name, voice_prediction, fake_probability, message):
    # Analyze the message for scam-related signals.
    text_result = analyze_text(message)

    # Create the voice result in the format expected by risk_scoring.py.
    voice_result = {
        "prediction": voice_prediction,
        "real_probability": 1 - fake_probability,
        "fake_probability": fake_probability
    }

    # Calculate the combined risk.
    risk_result = calculate_risk(voice_result, text_result)

    print("\n" + "=" * 60)
    print("TEST:", name)
    print("Voice prediction:", voice_prediction)
    print("Fake probability:", fake_probability)
    print("Message:", message)
    print("Risk score:", risk_result.get("risk_score"))
    print("Risk level:", risk_result.get("risk_level"))
    print("Reasons:", risk_result.get("reasons"))
    print("Action:", risk_result.get("action"))


def main():
    test_case(
        "Case 1 - Ordinary conversation",
        "real",
        0.02,
        "Hello, are you available for the meeting tomorrow?"
    )

    test_case(
        "Case 2 - OTP request",
        "real",
        0.02,
        "Please tell me the OTP sent to your phone."
    )

    test_case(
        "Case 3 - Multiple scam signals",
        "real",
        0.02,
        "I am calling from your bank. Tell me your OTP "
        "and transfer money through UPI immediately. "
        "Do not tell anyone."
    )

    test_case(
        "Case 4 - Suspicious voice prediction only",
        "fake",
        0.95,
        "Hello, are you available for the meeting tomorrow?"
    )

    test_case(
        "Case 5 - Fake prediction plus scam signals",
        "fake",
        0.95,
        "I am calling from your bank. Tell me your OTP "
        "and transfer money through UPI immediately. "
        "Do not tell anyone."
    )


if __name__ == "__main__":
    main()