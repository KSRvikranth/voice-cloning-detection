
from pathlib import Path
import csv
import soundfile as sf

DATASET_PATH = Path("dataset/raw/for-norm")
OUTPUT_PATH = Path("results/dataset_statistics.csv")

AUDIO_EXTENSIONS = {".wav", ".flac"}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not DATASET_PATH.exists():
    print("Dataset folder not found:", DATASET_PATH.resolve())
    raise SystemExit

print("Scanning dataset...", flush=True)

files = [
    f for f in DATASET_PATH.rglob("*")
    if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
]

print("Total audio files:", len(files), flush=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "sample_rate", "channels", "duration"])

    for i, file in enumerate(files, start=1):
        try:
            info = sf.info(str(file))
            duration = info.frames / info.samplerate

            writer.writerow([
                str(file),
                info.samplerate,
                info.channels,
                round(duration, 3)
            ])

        except Exception as e:
            print("Skipped:", file, "|", e, flush=True)

        if i % 500 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}", flush=True)

print("Finished! Saved to:", OUTPUT_PATH.resolve(), flush=True)