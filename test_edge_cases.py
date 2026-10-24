
from scripts.risk_scoring import calculate_risk
from scripts.social_engineering_detector import analyze_text


# Voice-model test results
real_voice = {
    "prediction": "real",
    "real_probability": 0.99,
    "fake_probability": 0.01
}

fake_voice = {
    "prediction": "fake",
    "real_probability": 0.01,
    "fake_probability": 0.99
}


# Edge cases to test
test_cases = [
    {
        "name": "1. Harmless conversation",
        "voice": real_voice,
        "text": "Hello, how are you? Have a nice day."
    },
    {
        "name": "2. Harmless OTP discussion",
        "voice": real_voice,
        "text": (
            "Never share your OTP with anyone. "
            "I learned about OTP safety today."
        )
    },
    {
        "name": "3. Scam wording not covered by patterns",
        "voice": real_voice,
        "text": (
            "Your account has a problem. "
            "Follow the instructions to protect it."
        )
    },
    {
        "name": "4. Fake voice, harmless text",
        "voice": fake_voice,
        "text": "The weather is nice today."
    },
    {
        "name": "5. Real voice, suspicious text",
        "voice": real_voice,
        "text": (
            "I am calling from your bank. "
            "Share your OTP immediately. "
            "Don't tell anyone."
        )
    },
    {
        "name": "6. Money and payment only",
        "voice": real_voice,
        "text": "Please send me money through UPI."
    },
    {
        "name": "7. Urgency without OTP",
        "voice": real_voice,
        "text": (
            "Your account will be blocked right now. "
            "Don't contact the bank."
        )
    }
]


print("\n===== DAY 70: EDGE-CASE TESTING =====")

for test in test_cases:
    text_result = analyze_text(test["text"])

    risk_result = calculate_risk(
        test["voice"],
        text_result
    )

    print("\n" + "=" * 45)
    print("Test:", test["name"])
    print("Transcript:", test["text"])
    print("Text status:", text_result["text_status"])
    print(
        "Detected categories:",
        text_result["detected_categories"]
    )
    print(
        "Strong text signals:",
        text_result["strong_text_signals"]
    )
    print("Voice prediction:", test["voice"]["prediction"])
    print("Risk score:", risk_result["risk_score"])
    print("Risk level:", risk_result["risk_level"])
    print("Reasons:", risk_result["reasons"])
    print("Suggested action:", risk_result["action"])

print("\n===== DAY 70 TESTS FINISHED =====")