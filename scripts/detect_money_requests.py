
import re
from pathlib import Path

# Read the transcription generated on Day 12
INPUT_FILE = Path("results/transcriptions/sample.txt")

# Save the detection result
OUTPUT_FILE = Path("results/money_detection.txt")

# Money-related phrases and patterns
MONEY_PATTERNS = {
    "money_request": [
        r"\bsend me money\b",
        r"\bsend money\b",
        r"\btransfer (?:the )?money\b",
        r"\btransfer (?:the )?amount\b",
        r"\bpay me\b",
        r"\bmake (?:a )?payment\b",
        r"\bneed money\b",
        r"\bi need cash\b",
        r"\bsend (?:the )?amount\b",
        r"\bdeposit (?:the )?money\b",
    ],

    "urgent_payment": [
        r"\burgent(?:ly)?\b",
        r"\bright now\b",
        r"\bas soon as possible\b",
        r"\bimmediately\b",
        r"\bdon't delay\b",
        r"\bdo it quickly\b",
    ],

    "payment_method": [
        r"\bupi\b",
        r"\bpaytm\b",
        r"\bgoogle pay\b",
        r"\bphonepe\b",
        r"\bbank transfer\b",
        r"\baccount number\b",
    ]
}


def detect_money_requests(text):
    text_lower = text.lower()

    detected = {}

    for category, patterns in MONEY_PATTERNS.items():
        matches = []

        for pattern in patterns:
            found = re.findall(pattern, text_lower)

            if found:
                matches.extend(found)

        detected[category] = sorted(set(matches))

    return detected


def main():
    if not INPUT_FILE.exists():
        print("Transcription file not found:", INPUT_FILE)
        print("Run the Day 12 transcription script first.")
        return

    text = INPUT_FILE.read_text(encoding="utf-8").strip()

    if not text:
        print("The transcription file is empty.")
        return

    results = detect_money_requests(text)

    print("\n===== MONEY REQUEST DETECTION =====")
    print("\nTranscription:")
    print(text)

    print("\nDetected signals:")

    found_any = False

    for category, matches in results.items():
        if matches:
            found_any = True
            print(f"{category}: {', '.join(matches)}")
        else:
            print(f"{category}: None")

    if found_any:
        print("\nMoney-related signals were detected.")
    else:
        print("\nNo configured money-related signals detected.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("TRANSCRIPTION:\n")
        file.write(text + "\n\n")
        file.write("DETECTED SIGNALS:\n")

        for category, matches in results.items():
            file.write(f"{category}: {matches}\n")

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()