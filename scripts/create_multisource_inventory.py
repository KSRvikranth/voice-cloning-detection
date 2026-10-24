from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLAAD_PATH = PROJECT_ROOT / "dataset" / "raw" / "MLAAD-tiny"
LIBRISPEECH_PATH = PROJECT_ROOT / "dataset" / "raw" / "LibriSpeech"

RESULTS_PATH = PROJECT_ROOT / "results" / "multisource_inventory.csv"


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac"
}


def get_audio_files(folder):

    if not folder.exists():
        return []

    files = []

    for file in folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(file)

    return files


def add_files(rows, files, dataset, label):

    for file in files:

        relative_path = file.relative_to(PROJECT_ROOT)

        rows.append({
            "file": str(relative_path),
            "dataset": dataset,
            "label": label,
            "language": "unknown",
            "speaker": "unknown"
        })


def main():

    print("=" * 70)
    print("MULTISOURCE DATASET INVENTORY")
    print("=" * 70)

    rows = []

    # -------------------------------------------------
    # MLAAD FAKE
    # -------------------------------------------------

    mlaad_fake = MLAAD_PATH / "fake"

    print("\nScanning MLAAD fake...")

    fake_files = get_audio_files(mlaad_fake)

    print("Found:", len(fake_files))

    add_files(
        rows,
        fake_files,
        "MLAAD",
        "fake"
    )

    # -------------------------------------------------
    # MLAAD ORIGINAL
    # -------------------------------------------------

    mlaad_original = MLAAD_PATH / "original"

    print("\nScanning MLAAD original...")

    original_files = get_audio_files(mlaad_original)

    print("Found:", len(original_files))

    add_files(
        rows,
        original_files,
        "MLAAD",
        "real"
    )

    # -------------------------------------------------
    # LIBRISPEECH
    # -------------------------------------------------

    print("\nScanning LibriSpeech...")

    libri_files = get_audio_files(LIBRISPEECH_PATH)

    print("Found:", len(libri_files))

    add_files(
        rows,
        libri_files,
        "LibriSpeech",
        "real"
    )

    # -------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "file",
        "dataset",
        "label",
        "language",
        "speaker"
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    fake_count = sum(
        1 for row in rows
        if row["label"] == "fake"
    )

    real_count = sum(
        1 for row in rows
        if row["label"] == "real"
    )

    print("\n" + "=" * 70)
    print("INVENTORY COMPLETE")
    print("=" * 70)

    print("Total audio files:", len(rows))
    print("Real files:", real_count)
    print("Fake files:", fake_count)

    print("\nInventory saved to:")

    print(RESULTS_PATH)


if __name__ == "__main__":
    main()