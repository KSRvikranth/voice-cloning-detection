from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INVENTORY_PATH = PROJECT_ROOT / "results" / "multisource_inventory.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "multisource_metadata.csv"


def extract_metadata(row):
    file_path = Path(row["file"])

    dataset = row["dataset"]
    label = row["label"]

    language = "unknown"
    speaker = "unknown"
    source = "unknown"

    parts = file_path.parts

    # ---------------------------------------------------------
    # MLAAD
    # ---------------------------------------------------------
    if dataset == "MLAAD":

        # Expected structure:
        # MLAAD-tiny/fake/de/CaroTTS/file.wav
        #
        # or
        # MLAAD-tiny/original/...

        try:
            # Find fake/original position
            if "fake" in parts:
                index = parts.index("fake")
            elif "original" in parts:
                index = parts.index("original")
            else:
                index = -1

            if index != -1:

                # language is normally immediately after fake/original
                if index + 1 < len(parts):
                    language = parts[index + 1]

                # TTS/source system normally follows language
                if index + 2 < len(parts):
                    source = parts[index + 2]

        except Exception:
            pass

        # MLAAD does not necessarily expose a human speaker ID
        # in the directory structure, so keep speaker unknown.
        speaker = "unknown"

    # ---------------------------------------------------------
    # LIBRISPEECH
    # ---------------------------------------------------------
    elif dataset == "LibriSpeech":

        # Expected:
        # LibriSpeech/test-clean/1089/134686/file.flac

        try:
            if "test-clean" in parts:

                index = parts.index("test-clean")

                # Speaker ID
                if index + 1 < len(parts):
                    speaker = parts[index + 1]

                # LibriSpeech test-clean is English
                language = "en"

                source = "LibriSpeech"

        except Exception:
            pass

    return {
        "file": row["file"],
        "dataset": dataset,
        "label": label,
        "language": language,
        "speaker": speaker,
        "source": source
    }


def main():

    print("=" * 70)
    print("ENRICHING MULTISOURCE METADATA")
    print("=" * 70)

    if not INVENTORY_PATH.exists():
        print("\nERROR: Inventory file not found:")
        print(INVENTORY_PATH)
        return

    rows = []

    with open(
        INVENTORY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(extract_metadata(row))

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "file",
            "dataset",
            "label",
            "language",
            "speaker",
            "source"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total = len(rows)

    real_count = sum(
        1 for row in rows
        if row["label"] == "real"
    )

    fake_count = sum(
        1 for row in rows
        if row["label"] == "fake"
    )

    mlaad_count = sum(
        1 for row in rows
        if row["dataset"] == "MLAAD"
    )

    libri_count = sum(
        1 for row in rows
        if row["dataset"] == "LibriSpeech"
    )

    languages = {}

    for row in rows:

        language = row["language"]

        languages[language] = (
            languages.get(language, 0) + 1
        )

    print("\n" + "=" * 70)
    print("METADATA COMPLETE")
    print("=" * 70)

    print("\nTotal files:", total)
    print("Real files:", real_count)
    print("Fake files:", fake_count)

    print("\nDataset counts:")
    print("MLAAD:", mlaad_count)
    print("LibriSpeech:", libri_count)

    print("\nLanguages:")

    for language, count in sorted(languages.items()):
        print(f"{language}: {count}")

    print("\nMetadata saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()