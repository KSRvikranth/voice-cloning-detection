
from scripts.risk_scoring import calculate_risk
from scripts.social_engineering_detector import analyze_text


# Test 1: Harmless conversation
harmless_text = (
    "Hello, how are you? "
    "I will talk to you tomorrow."
)

# Test 2: Money and UPI mentioned, but no strong warning signals
payment_text = (
    "Can you send me money through UPI?"
)

# Test 3: Scam-like conversation
scam_text = (
    "I am calling from your bank. "
    "Share your OTP and don't tell anyone. "
    "Your account will be blocked right now. "
    "Send me money using UPI."
)


# Two different voice-model results
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


# Define test cases and expected results
test_cases = [
    ("Harmless conversation", real_voice, harmless_text, 0, "LOW"),
    ("Payment mention only", real_voice, payment_text, 5, "LOW"),
    ("Real voice, scam-like text", real_voice, scam_text, 80, "HIGH"),
    ("Fake voice, harmless text", fake_voice, harmless_text, 20, "LOW"),
    ("Fake voice, scam-like text", fake_voice, scam_text, 100, "HIGH"),
]


print("\n===== DAY 69: RISK-SCORING TESTS =====")

passed = 0

for name, voice_result, transcript, expected_score, expected_level in test_cases:

    # Detect signals in the transcript
    text_result = analyze_text(transcript)

    # Calculate the combined risk score
    result = calculate_risk(voice_result, text_result)

    actual_score = result["risk_score"]
    actual_level = result["risk_level"]

    print("\nTest:", name)
    print("Expected score:", expected_score)
    print("Actual score:", actual_score)
    print("Risk level:", actual_level)
    print("Detected signals:", result["signals"])

    # Check whether the result matches expectations
    if actual_score == expected_score and actual_level == expected_level:
        print("Result: PASS")
        passed += 1
    else:
        print("Result: FAIL")

print("\n==============================")
print("Tests passed:", passed, "out of", len(test_cases))
print("==============================")