import os
import librosa
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CSV_FILES = [
    "results/multisource_train.csv",
    "results/multisource_validation.csv",
    "results/multisource_test.csv"
]

TARGET_SAMPLE_RATE = 16000


# ============================================================
# CHECK ONE CSV FILE
# ============================================================

def check_csv(csv_path):

    print()
    print("=" * 70)
    print("Checking:", csv_path)
    print("=" * 70)

    df = pd.read_csv(csv_path)

    print("Total files:", len(df))

    successful = 0
    failed = 0

    real_count = 0
    fake_count = 0

    failed_files = []

    sample_rates = {}

    for index, row in df.iterrows():

        audio_path = row["file"]

        label = str(row["label"]).lower()

        # ----------------------------------------------------
        # Count labels
        # ----------------------------------------------------

        if label == "real":
            real_count += 1

        elif label == "fake":
            fake_count += 1

        # ----------------------------------------------------
        # Check whether file exists
        # ----------------------------------------------------

        if not os.path.exists(audio_path):

            failed += 1

            failed_files.append(
                (audio_path, "FILE NOT FOUND")
            )

            continue

        # ----------------------------------------------------
        # Try loading audio
        # ----------------------------------------------------

        try:

            audio, sr = librosa.load(
                audio_path,
                sr=TARGET_SAMPLE_RATE,
                mono=True
            )

            # Make sure something was loaded

            if audio is None or len(audio) == 0:

                failed += 1

                failed_files.append(
                    (audio_path, "EMPTY AUDIO")
                )

                continue

            successful += 1

            # Track resulting sample rate

            sample_rates[sr] = (
                sample_rates.get(sr, 0) + 1
            )

        except Exception as e:

            failed += 1

            failed_files.append(
                (audio_path, str(e))
            )


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("Real files:", real_count)
    print("Fake files:", fake_count)

    print()
    print("Successfully loaded:", successful)
    print("Failed:", failed)

    print()
    print("Sample rates after loading:")

    for sr, count in sample_rates.items():

        print(
            f"  {sr} Hz -> {count} files"
        )


    # ========================================================
    # FAILED FILES
    # ========================================================

    if failed_files:

        print()
        print("-" * 70)
        print("FAILED FILES")
        print("-" * 70)

        for path, reason in failed_files:

            print()
            print("File:", path)
            print("Reason:", reason)

    else:

        print()
        print("No failed files found.")


    return {
        "total": len(df),
        "successful": successful,
        "failed": failed,
        "real": real_count,
        "fake": fake_count
    }


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("DAY 47 - TRAINING AUDIO VALIDATION")
print("=" * 70)


all_results = []


for csv_file in CSV_FILES:

    if not os.path.exists(csv_file):

        print()
        print("CSV NOT FOUND:", csv_file)

        continue

    result = check_csv(csv_file)

    result["csv"] = csv_file

    all_results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DAY 47 FINAL SUMMARY")
print("=" * 70)

for result in all_results:

    print()
    print(result["csv"])

    print(
        "Total:",
        result["total"]
    )

    print(
        "Successful:",
        result["successful"]
    )

    print(
        "Failed:",
        result["failed"]
    )

    print(
        "Real:",
        result["real"]
    )

    print(
        "Fake:",
        result["fake"]
    )


print()
print("DAY 47 COMPLETE")