
import re
from pathlib import Path

INPUT_FILE = Path("results/transcriptions/money_test.txt")
OUTPUT_FILE = Path("results/impersonation_detection.txt")

PATTERNS = {
    "authority_claim": [
        r"\bi am (?:from|with) the police\b",
        r"\bi am (?:from|with) your bank\b",
        r"\bi am a bank (?:officer|employee|manager)\b",
        r"\bi am calling from the bank\b",
        r"\bi am calling from the police\b",
        r"\bi am from the government\b",
    ],

    "family_or_friend_claim": [
        r"\bit is me\b",
        r"\bi am your (?:son|daughter|brother|sister|friend)\b",
        r"\bi am your family member\b",
    ],

    "workplace_claim": [
        r"\bi am your (?:boss|manager)\b",
        r"\bi am calling from (?:the )?office\b",
        r"\bi am from the company\b",
    ],

    "identity_claim": [
        r"\bthis is (?:your )?(?:bank manager|manager|officer)\b",
        r"\bmy name is\b",
        r"\bi am calling on behalf of\b",
    ],
}


def detect_impersonation(text):
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

    results = detect_impersonation(text)

    print("\n===== IMPERSONATION DETECTION =====")
    print("Transcription:", text)

    found_any = False

    for category, matches in results.items():
        if matches:
            found_any = True
            print(f"{category}: {matches}")
        else:
            print(f"{category}: None")

    if found_any:
        print("\nIdentity or authority claims detected.")
        print("The claims have not been verified.")
    else:
        print("\nNo configured impersonation phrases detected.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("TRANSCRIPTION:\n")
        file.write(text + "\n\n")
        file.write("IMPERSONATION SIGNALS:\n")

        for category, matches in results.items():
            file.write(f"{category}: {matches}\n")

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()