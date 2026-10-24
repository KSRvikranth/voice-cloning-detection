from pathlib import Path
import librosa
import soundfile as sf

INPUT_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("dataset/preprocessed")

SAMPLE_RATE = 16000
DURATION = 5

SAMPLES = SAMPLE_RATE * DURATION


def process_file(input_file, output_file):

    audio, sr = librosa.load(
        input_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < SAMPLES:

        audio = librosa.util.fix_length(
            audio,
            size=SAMPLES
        )

    else:

        audio = audio[:SAMPLES]

    sf.write(
        output_file,
        audio,
        SAMPLE_RATE
    )


def process_category(category):

    input_dir = INPUT_DIR / category
    output_dir = OUTPUT_DIR / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = list(input_dir.glob("*.wav"))

    print(category.upper())
    print("Files:", len(files))

    for i, file in enumerate(files, start=1):

        output_file = output_dir / file.name

        try:

            process_file(
                file,
                output_file
            )

        except Exception as e:

            print("Error:", file)
            print(e)

        if i % 500 == 0 or i == len(files):

            print(
                f"Processed {i}/{len(files)}"
            )


process_category("real")
process_category("fake")

print()
print("PREPROCESSING COMPLETE")