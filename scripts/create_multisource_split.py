from pathlib import Path
import csv
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "results" / "multisource_metadata.csv"

TRAIN_PATH = PROJECT_ROOT / "results" / "multisource_train.csv"
VALIDATION_PATH = PROJECT_ROOT / "results" / "multisource_validation.csv"
TEST_PATH = PROJECT_ROOT / "results" / "multisource_test.csv"


SEED = 42


def write_csv(path, rows):

    if not rows:
        print("WARNING: No rows for", path)
        return

    fieldnames = [
        "file",
        "dataset",
        "label",
        "language",
        "speaker",
        "source"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def split_groups(groups, train_ratio=0.70, validation_ratio=0.15):

    groups = list(groups)

    random.shuffle(groups)

    total = len(groups)

    train_end = int(total * train_ratio)

    validation_end = int(
        total * (train_ratio + validation_ratio)
    )

    train_groups = set(groups[:train_end])

    validation_groups = set(
        groups[train_end:validation_end]
    )

    test_groups = set(
        groups[validation_end:]
    )

    return (
        train_groups,
        validation_groups,
        test_groups
    )


def main():

    print("=" * 70)
    print("CREATING MULTISOURCE DATASET SPLIT")
    print("=" * 70)

    if not INPUT_PATH.exists():

        print("\nERROR: Metadata file not found:")
        print(INPUT_PATH)

        return

    random.seed(SEED)

    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------

    rows = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    print("\nTotal records:", len(rows))

    # ---------------------------------------------------------
    # Separate datasets
    # ---------------------------------------------------------

    mlaad_rows = [
        row for row in rows
        if row["dataset"] == "MLAAD"
    ]

    libri_rows = [
        row for row in rows
        if row["dataset"] == "LibriSpeech"
    ]

    print("MLAAD records:", len(mlaad_rows))
    print("LibriSpeech records:", len(libri_rows))

    # ---------------------------------------------------------
    # MLAAD
    #
    # Use source + language as grouping unit.
    # ---------------------------------------------------------

    mlaad_groups = {}

    for row in mlaad_rows:

        group = (
            row["label"],
            row["language"],
            row["source"]
        )

        mlaad_groups.setdefault(
            group,
            []
        ).append(row)

    # ---------------------------------------------------------
    # LibriSpeech
    #
    # Use speaker as grouping unit.
    # ---------------------------------------------------------

    libri_groups = {}

    for row in libri_rows:

        group = (
            row["label"],
            row["speaker"]
        )

        libri_groups.setdefault(
            group,
            []
        ).append(row)

    print("\nMLAAD groups:", len(mlaad_groups))
    print("LibriSpeech speaker groups:", len(libri_groups))

    # ---------------------------------------------------------
    # Split MLAAD groups
    # ---------------------------------------------------------

    mlaad_group_keys = list(mlaad_groups.keys())

    (
        mlaad_train_groups,
        mlaad_validation_groups,
        mlaad_test_groups
    ) = split_groups(mlaad_group_keys)

    mlaad_train = []
    mlaad_validation = []
    mlaad_test = []

    for group, group_rows in mlaad_groups.items():

        if group in mlaad_train_groups:

            mlaad_train.extend(group_rows)

        elif group in mlaad_validation_groups:

            mlaad_validation.extend(group_rows)

        else:

            mlaad_test.extend(group_rows)

    # ---------------------------------------------------------
    # Split LibriSpeech speakers
    # ---------------------------------------------------------

    libri_group_keys = list(libri_groups.keys())

    (
        libri_train_groups,
        libri_validation_groups,
        libri_test_groups
    ) = split_groups(libri_group_keys)

    libri_train = []
    libri_validation = []
    libri_test = []

    for group, group_rows in libri_groups.items():

        if group in libri_train_groups:

            libri_train.extend(group_rows)

        elif group in libri_validation_groups:

            libri_validation.extend(group_rows)

        else:

            libri_test.extend(group_rows)

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    train_rows = mlaad_train + libri_train
    validation_rows = mlaad_validation + libri_validation
    test_rows = mlaad_test + libri_test

    # Shuffle final datasets
    random.shuffle(train_rows)
    random.shuffle(validation_rows)
    random.shuffle(test_rows)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    write_csv(
        TRAIN_PATH,
        train_rows
    )

    write_csv(
        VALIDATION_PATH,
        validation_rows
    )

    write_csv(
        TEST_PATH,
        test_rows
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def show_stats(name, data):

        real_count = sum(
            1 for row in data
            if row["label"] == "real"
        )

        fake_count = sum(
            1 for row in data
            if row["label"] == "fake"
        )

        print(f"\n{name}")
        print("-" * 40)
        print("Total:", len(data))
        print("Real:", real_count)
        print("Fake:", fake_count)

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    show_stats("TRAINING SET", train_rows)
    show_stats("VALIDATION SET", validation_rows)
    show_stats("TEST SET", test_rows)

    print("\nFiles created:")

    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)

    print("\nRandom seed:", SEED)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()