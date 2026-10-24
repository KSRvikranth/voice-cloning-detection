
# scripts/explainable_warning.py

def create_explanation(risk_result):
    """
    Convert the Day 20 risk result into a
    readable warning report.
    """

    score = risk_result.get("risk_score", 0)
    level = risk_result.get("risk_level", "UNKNOWN")
    reasons = risk_result.get("reasons", [])
    signals = risk_result.get("signals", [])
    action = risk_result.get("action", "")

    # ----------------------------------
    # 1. Explain the overall risk level
    # ----------------------------------
    if level == "HIGH":
        summary = (
            "Multiple or stronger warning signals were "
            "detected. Verify the caller independently."
        )

    elif level == "MEDIUM":
        summary = (
            "Some warning signals were detected. "
            "Be cautious before sharing sensitive "
            "information or sending money."
        )

    elif level == "LOW":
        summary = (
            "Few or no configured warning signals "
            "were detected. This does not guarantee "
            "that the call is safe."
        )

    else:
        summary = (
            "The risk level could not be interpreted."
        )

    # ----------------------------------
    # 2. Explain individual signals
    # ----------------------------------
    signal_explanations = {
        "money_request": (
            "A request involving money was detected. "
            "This can also happen in normal conversations."
        ),

        "payment_method": (
            "A payment method such as UPI was mentioned. "
            "A payment mention alone is not proof of fraud."
        ),

        "otp_request": (
            "The conversation appears to request an OTP. "
            "Do not share OTPs with callers."
        ),

        "impersonation": (
            "The conversation contains a possible "
            "impersonation signal. Verify the caller "
            "using an independently obtained contact."
        ),

        "pressure": (
            "Urgency or secrecy language was detected. "
            "Pause and verify before taking action."
        ),

        "voice_model_fake_prediction": (
            "The voice model predicted potentially "
            "synthetic audio. This is a model prediction, "
            "not proof of caller identity or fraud."
        )
    }

    explanations = []

    for signal in signals:
        explanation = signal_explanations.get(
            signal,
            f"Detected signal: {signal.replace('_', ' ')}."
        )

        explanations.append({
            "signal": signal,
            "explanation": explanation
        })

    # ----------------------------------
    # 3. Include the original reasons
    # ----------------------------------
    return {
        "risk_score": score,
        "risk_level": level,
        "summary": summary,
        "detected_signals": signals,
        "signal_explanations": explanations,
        "scoring_reasons": reasons,
        "suggested_action": action,
        "disclaimer": (
            "This is a heuristic warning system. "
            "It does not prove that a call is safe "
            "or fraudulent."
        )
    }


# --------------------------------------
# Standalone test
# --------------------------------------
if __name__ == "__main__":

    sample_result = {
        "risk_score": 75,
        "risk_level": "HIGH",
        "reasons": [
            "Detected concerning conversation signal: otp request.",
            "Detected concerning conversation signal: impersonation.",
            "Detected concerning conversation signal: pressure."
        ],
        "signals": [
            "impersonation",
            "otp_request",
            "pressure"
        ],
        "action": (
            "Show a strong warning. Verify the caller "
            "through a trusted, independent channel."
        )
    }

    report = create_explanation(sample_result)

    print("\n===== EXPLAINABLE WARNING =====")
    print("Risk score:", report["risk_score"])
    print("Risk level:", report["risk_level"])
    print("Summary:", report["summary"])

    print("\nDetected signals:")

    for item in report["signal_explanations"]:
        print("-", item["signal"])
        print(" ", item["explanation"])

    print("\nSuggested action:")
    print(report["suggested_action"])

    print("\nDisclaimer:")
    print(report["disclaimer"])