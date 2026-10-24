
# scripts/risk_scoring.py

def calculate_risk(voice_result, text_result):
    """
    Combine voice-model output and social-engineering signals.
    Returns an explainable risk score from 0 to 100.
    """

    score = 0
    reasons = []
    signals = []

    # ----------------------------------
    # 1. Voice model signal
    # ----------------------------------
    prediction = str(
        voice_result.get("prediction", "unknown")
    ).lower()

    fake_probability = voice_result.get(
        "fake_probability", 0
    )

    try:
        fake_probability = float(fake_probability)
    except (TypeError, ValueError):
        fake_probability = 0

    # Treat this as a limited model signal,
    # not proof that the caller is a scammer.
    if prediction == "fake":
        score += 20
        reasons.append(
            "Voice model predicted potentially synthetic audio."
        )
        signals.append("voice_model_fake_prediction")

    # ----------------------------------
    # 2. Read text detector output
    # ----------------------------------
    detected_categories = text_result.get(
        "detected_categories", []
    )

    if not isinstance(detected_categories, list):
        detected_categories = []

    categories = {
        str(category).lower()
        for category in detected_categories
    }

    strong_text_signals = text_result.get(
        "strong_text_signals", []
    )

    if not isinstance(strong_text_signals, list):
        strong_text_signals = []

    strong_signals = {
        str(signal).lower()
        for signal in strong_text_signals
    }

    # ----------------------------------
    # 3. Weak signals
    # ----------------------------------
    # Money/payment alone is not enough
    # to label a conversation high risk.

    weak_categories = {
        "money_request",
        "payment_method"
    }

    weak_found = categories.intersection(
        weak_categories
    )

    if weak_found:
        score += 5
        reasons.append(
            "Money or payment was mentioned. "
            "This alone is not proof of a scam."
        )
        signals.extend(sorted(weak_found))

    # ----------------------------------
    # 4. Stronger text signals
    # ----------------------------------
    signal_weights = {
        "otp_request": 30,
        "impersonation": 25,
        "pressure": 20
    }

    for signal, weight in signal_weights.items():
        if signal in categories or signal in strong_signals:
            score += weight
            reasons.append(
                f"Detected concerning conversation signal: "
                f"{signal.replace('_', ' ')}."
            )
            signals.append(signal)

    # ----------------------------------
    # 5. Limit score to 100
    # ----------------------------------
    score = min(score, 100)

    # ----------------------------------
    # 6. Assign risk level
    # ----------------------------------
    if score >= 60:
        risk_level = "HIGH"
        action = (
            "Show a strong warning. Verify the caller "
            "through a trusted, independent channel."
        )

    elif score >= 30:
        risk_level = "MEDIUM"
        action = (
            "Show a caution message. Check the request "
            "before sharing information or sending money."
        )

    else:
        risk_level = "LOW"
        action = (
            "No strong warning signals were found. "
            "This does not guarantee the call is safe."
        )

    # ----------------------------------
    # 7. Return explainable result
    # ----------------------------------
    return {
        "risk_score": score,
        "risk_level": risk_level,
        "reasons": reasons,
        "signals": sorted(set(signals)),
        "action": action,
        "voice_prediction": prediction,
        "fake_probability": fake_probability
    }


# --------------------------------------
# Simple standalone test
# --------------------------------------
if __name__ == "__main__":

    test_voice = {
        "prediction": "real",
        "real_probability": 1.0,
        "fake_probability": 0.0
    }

    test_text = {
        "detected_categories": [
            "money_request",
            "payment_method"
        ],
        "strong_text_signals": [],
        "text_status": "PAYMENT_MENTION_ONLY"
    }

    result = calculate_risk(
        test_voice,
        test_text
    )

    print("\n===== RISK SCORING TEST =====")
    print(result)