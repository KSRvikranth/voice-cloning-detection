
import re
from pathlib import Path

INPUT_FILE = Path("results/transcriptions/money_test.txt")
OUTPUT_FILE = Path("results/payment_detection.txt")

PAYMENT_PATTERNS = {
    "payment_apps": [
        r"\bupi\b",
        r"\bpaytm\b",
        r"\bgoogle pay\b",
        r"\bgpay\b",
        r"\bphonepe\b",
        r"\bphone pe\b",
        r"\bbhim\b",
    ],

    "payment_details": [
        r"\bupi id\b",
        r"\bqr code\b",
        r"\bscan (?:the )?qr\b",
        r"\bbank account\b",
        r"\baccount number\b",
        r"\bifsc\b",
        r"\bpayment link\b",
        r"\btransaction id\b",
        r"\butr number\b",
    ],

    "payment_actions": [
        r"\bmake (?:a )?payment\b",
        r"\bpay (?:the )?amount\b",
        r"\bsend (?:the )?amount\b",
        r"\btransfer (?:the )?money\b",
        r"\bscan and pay\b",
        r"\bclick (?:the )?payment link\b",
    ],
}


def detect_payment_signals(text):
    text = text.lower()
    results = {}

    for category, patterns in PAYMENT_PATTERNS.items():
        matches = []

        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)

        results[category] = sorted(set(matches))

    return results


def main():
    if not INPUT_FILE.exists():
        print("Input transcription not found:", INPUT_FILE)
        print("Check the INPUT_FILE path.")
        return

    text = INPUT_FILE.read_text(encoding="utf-8").strip()

    if not text:
        print("Transcription file is empty.")
        return

    results = detect_payment_signals(text)

    print("\n===== PAYMENT SIGNAL DETECTION =====")
    print("\nTranscription:")
    print(text)

    any_found = False

    for category, matches in results.items():
        if matches:
            any_found = True
            print(f"{category}: {', '.join(matches)}")
        else:
            print(f"{category}: None")

    if any_found:
        print("\nPayment-related language detected.")
        print("This is a signal, not proof of fraud.")
    else:
        print("\nNo configured payment signals detected.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("TRANSCRIPTION:\n")
        file.write(text + "\n\n")
        file.write("PAYMENT SIGNALS:\n")

        for category, matches in results.items():
            file.write(f"{category}: {matches}\n")

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()