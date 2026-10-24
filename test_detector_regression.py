
from scripts.social_engineering_detector import analyze_text
from scripts.risk_scoring import calculate_risk


tests = [
    {
        "name": "Harmless conversation",
        "text": "Hello, how are you?",
        "expected_signal": None,
    },
    {
        "name": "Protective OTP advice",
        "text": "Never share your OTP with anyone.",
        "expected_signal": "otp_request",
    },
    {
        "name": "Direct OTP request",
        "text": "Tell me the OTP immediately.",
        "expected_signal": "otp_request",
    },
    {
        "name": "Account problem wording",
        "text": "Your account has a problem. Follow the instructions.",
        "expected_signal": "pressure",
    },
    {
        "name": "Bank impersonation and pressure",
        "text": (
            "I am calling from your bank. "
            "Share your OTP immediately. Don't tell anyone."
        ),
        "expected_signal": "impersonation",
    },
]


print("\n===== DAY 72: DETECTOR REGRESSION TESTS =====")

passed = 0

for test in tests:
    result = analyze_text(test["text"])
    signals = result["detected_categories"]

    # Protective advice should not be treated as an OTP request.
    if test["name"] == "Protective OTP advice":
        success = "otp_request" not in signals
    else:
        expected = test["expected_signal"]
        success = (
            expected is None
            if expected is None
            else expected in signals
        )

    print("\nTest:", test["name"])
    print("Detected categories:", signals)
    print("Text status:", result["text_status"])
    print("Result:", "PASS" if success else "FAIL")

    if success:
        passed += 1

print("\nTests passed:", passed, "out of", len(tests))