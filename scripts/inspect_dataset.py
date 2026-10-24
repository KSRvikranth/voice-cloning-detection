import os
from pathlib import Path

DATASET_PATH = Path("dataset/raw/for-norm")

audio_extensions = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a"
}

if not DATASET_PATH.exists():
    print("Dataset folder not found!")
    print("Expected location:")
    print(DATASET_PATH)
    exit()

total_files = 0
audio_files = 0

print("=" * 50)
print("FoR DATASET INSPECTION")
print("=" * 50)

for file in DATASET_PATH.rglob("*"):

    if file.is_file():

        total_files += 1

        if file.suffix.lower() in audio_extensions:
            audio_files += 1

print("Total files:", total_files)
print("Audio files:", audio_files)

print("\nTop-level folders:")

for item in DATASET_PATH.iterdir():

    if item.is_dir():
        print("  ", item.name)

print("\nDataset inspection completed.")