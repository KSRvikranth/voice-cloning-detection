from pathlib import Path
import shutil

SOURCE_DIR = Path("dataset/raw/for-norm")

OUTPUT_DIR = Path("dataset/processed")

REAL_DIR = OUTPUT_DIR / "real"
FAKE_DIR = OUTPUT_DIR / "fake"

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a"
}

REAL_KEYWORDS = {
    "real",
    "bonafide",
    "genuine"
}

FAKE_KEYWORDS = {
    "fake",
    "spoof",
    "synthetic",
    "generated"
}


def find_class(file):
    path_text = str(file).lower()

    for keyword in REAL_KEYWORDS:
        if keyword in path_text:
            return "real"

    for keyword in FAKE_KEYWORDS:
        if keyword in path_text:
            return "fake"

    return None


def main():

    if not SOURCE_DIR.exists():
        print("Dataset folder not found:")
        print(SOURCE_DIR.resolve())
        return

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        file
        for file in SOURCE_DIR.rglob("*")
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print("Audio files found:", len(files))

    real_count = 0
    fake_count = 0
    skipped = 0

    for i, file in enumerate(files, start=1):

        category = find_class(file)

        if category is None:
            skipped += 1
            continue

        if category == "real":
            destination_dir = REAL_DIR
            real_count += 1
        else:
            destination_dir = FAKE_DIR
            fake_count += 1

        destination = destination_dir / file.name

        # Avoid overwriting files with the same name
        counter = 1

        while destination.exists():

            destination = (
                destination_dir /
                f"{file.stem}_{counter}{file.suffix}"
            )

            counter += 1

        shutil.copy2(file, destination)

        if i % 500 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}")

    print()
    print("=" * 50)
    print("DATASET ORGANIZATION COMPLETE")
    print("=" * 50)

    print("Real files:", real_count)
    print("Fake files:", fake_count)
    print("Skipped:", skipped)

    print()
    print("Real folder:", REAL_DIR.resolve())
    print("Fake folder:", FAKE_DIR.resolve())


if __name__ == "__main__":
    main()