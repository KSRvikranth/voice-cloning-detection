
import re


PATTERNS = {
    "money_request": [
        r"\bsend me money\b",
        r"\bsend money\b",
        r"\btransfer (?:the )?money\b",
        r"\bpay me\b",
        r"\bmake (?:a )?payment\b",
        r"\bi need cash\b",
        r"\bsend (?:the )?amount\b",
    ],

    "otp_request": [
        r"\botp\b",
        r"\bone.time password\b",
        r"\bverification code\b",
        r"\bshare (?:the )?code\b",
        r"\btell me (?:the )?code\b",
    ],

    "payment_method": [
        r"\bupi\b",
        r"\bpaytm\b",
        r"\bgoogle pay\b",
        r"\bgpay\b",
        r"\bphonepe\b",
        r"\bqr code\b",
        r"\bpayment link\b",
    ],

    "impersonation": [
        r"\bi am calling from your bank\b",
        r"\bi am from the police\b",
        r"\bi am a bank officer\b",
        r"\bi am your (?:son|daughter|brother|sister)\b",
        r"\bi am your manager\b",
    ],

    "pressure": [
        r"\bright now\b",
        r"\bimmediately\b",
        r"\bdon't tell anyone\b",
        r"\bkeep this secret\b",
        r"\byour account will be blocked\b",
        r"\byou will be arrested\b",
        r"\bdon't contact the bank\b",
        r"\byour account has a problem\b",
        r"\byour account is at risk\b",
        r"\bverify your account\b",
        r"\bact now\b",
    ],
}


# These phrases usually indicate a warning or safety discussion,
# rather than a request to reveal an OTP.
PROTECTIVE_OTP_PATTERNS = [
    r"\bnever share (?:your )?otp\b",
    r"\bdo not share (?:your )?otp\b",
    r"\bdon't share (?:your )?otp\b",
    r"\bnever tell anyone (?:your )?otp\b",
    r"\bkeep your otp private\b",
]


def analyze_text(text):
    text = str(text).lower()
    results = {}

    protective_otp = any(
        re.search(pattern, text)
        for pattern in PROTECTIVE_OTP_PATTERNS
    )

    for category, patterns in PATTERNS.items():
        matches = []

        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)

        # Do not treat a protective OTP warning as an OTP request
        # unless the transcript also contains a direct code request.
        if category == "otp_request" and protective_otp:
            direct_request_patterns = [
                r"\bshare (?:the )?otp\b",
                r"\btell me (?:the )?otp\b",
                r"\bsend (?:me )?(?:the )?otp\b",
                r"\bgive me (?:the )?otp\b",
            ]

            direct_request = any(
                re.search(pattern, text)
                for pattern in direct_request_patterns
            )

            if not direct_request:
                matches = []

        results[category] = sorted(set(matches))

    detected_categories = [
        category
        for category, matches in results.items()
        if matches
    ]

    results["detected_categories"] = detected_categories
    results["category_count"] = len(detected_categories)

    other_signals = [
        "otp_request",
        "impersonation",
        "pressure",
    ]

    results["strong_text_signals"] = [
        category
        for category in other_signals
        if results[category]
    ]

    if len(results["strong_text_signals"]) >= 2:
        status = "HIGH_CONCERN"
    elif len(results["strong_text_signals"]) == 1:
        status = "NEEDS_ATTENTION"
    elif results["money_request"] or results["payment_method"]:
        status = "PAYMENT_MENTION_ONLY"
    else:
        status = "NO_CONFIGURED_WARNING_SIGNAL"

    results["text_status"] = status

    return results


if __name__ == "__main__":
    samples = [
        "Never share your OTP with anyone.",
        "I am calling from your bank. Share your OTP immediately.",
        "Your account has a problem. Follow the instructions to protect it.",
    ]

    for sample in samples:
        print("\nTranscript:", sample)
        print(analyze_text(sample))