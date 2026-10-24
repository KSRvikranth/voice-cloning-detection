
import re
from pathlib import Path

INPUT_FILE = Path("results/transcriptions/money_test.txt")
OUTPUT_FILE = Path("results/pressure_detection.txt")

PATTERNS = {
    "urgency": [
        r"\bright now\b",
        r"\bimmediately\b",
        r"\burgently\b",
        r"\bwithin \d+ minutes?\b",
        r"\bdo it quickly\b",
        r"\bdon't delay\b",
        r"\bas soon as possible\b",
    ],

    "secrecy": [
        r"\bdon't tell anyone\b",
        r"\bkeep this secret\b",
        r"\bdon't tell your family\b",
        r"\bkeep it between us\b",
        r"\bdon't inform anyone\b",
    ],

    "pressure_or_threat": [
        r"\byou must do this\b",
        r"\bdo it now\b",
        r"\byou will be arrested\b",
        r"\byour account will be blocked\b",
        r"\byour account will be suspended\b",
        r"\blast chance\b",
    ],

    "discouraging_verification": [
        r"\bdon't call back\b",
        r"\bdon't contact the bank\b",
        r"\bdon't verify\b",
        r"\bno need to check\b",
    ],
}


def detect_pressure(text):
    text = text.lower()
    results = {}

    for category, patterns in PATTERNS.items():
        matches = []

        for pattern in patterns:
            matches.extend(re.findall(pattern, text))

        results[category] = sorted(set(matches))

    return results


def main():
    if not INPUT_FILE.exists():
        print("Transcription file not found:", INPUT_FILE)
        return

    text = INPUT_FILE.read_text(encoding="utf-8").strip()

    if not text:
        print("Transcription is empty.")
        return

    results = detect_pressure(text)

    print("\n===== PRESSURE DETECTION =====")
    print("Transcription:", text)

    found_any = False

    for category, matches in results.items():
        if matches:
            found_any = True
            print(f"{category}: {matches}")
        else:
            print(f"{category}: None")

    if found_any:
        print("\nPressure-related language detected.")
    else:
        print("\nNo configured pressure phrases detected.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("TRANSCRIPTION:\n")
        file.write(text + "\n\n")
        file.write("PRESSURE SIGNALS:\n")

        for category, matches in results.items():
            file.write(f"{category}: {matches}\n")

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()