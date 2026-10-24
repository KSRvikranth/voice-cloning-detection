from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INVENTORY_PATH = PROJECT_ROOT / "results" / "multisource_inventory.csv"


def inspect_paths():

    print("=" * 70)
    print("MULTISOURCE METADATA INSPECTION")
    print("=" * 70)

    if not INVENTORY_PATH.exists():
        print("Inventory file not found:")
        print(INVENTORY_PATH)
        return

    with open(
        INVENTORY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print("\nTotal records:", len(rows))

    print("\n" + "-" * 70)
    print("SAMPLE MLAAD FILES")
    print("-" * 70)

    count = 0

    for row in rows:

        if row["dataset"] == "MLAAD":

            print(row["file"])

            count += 1

            if count >= 10:
                break

    print("\n" + "-" * 70)
    print("SAMPLE LIBRISPEECH FILES")
    print("-" * 70)

    count = 0

    for row in rows:

        if row["dataset"] == "LibriSpeech":

            print(row["file"])

            count += 1

            if count >= 10:
                break


if __name__ == "__main__":
    inspect_paths()