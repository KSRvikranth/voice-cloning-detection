
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

OUTPUT = RESULTS / "day56_final_evaluation_report.csv"


def add_result(rows, name, accuracy, fake_recall, fake_f1,
               roc_auc, test_files, notes):
    rows.append({
        "experiment": name,
        "accuracy": accuracy,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "roc_auc": roc_auc,
        "files_evaluated": test_files,
        "notes": notes
    })


def main():
    rows = []

    # V1: existing held-out test evaluation.
    add_result(
        rows,
        "V1 - Multisource held-out test",
        0.875554,
        0.8125,
        0.852645,
        0.9279,
        2708,
        "Held-out split; inspect split limitations"
    )

    # V2: existing held-out test evaluation.
    add_result(
        rows,
        "V2 - Multisource held-out test",
        0.807976,
        0.665,
        0.754253,
        0.891221,
        2708,
        "Same test split as V1"
    )

    # External results from Day 52.
    add_result(
        rows,
        "V1 - External, trimmed",
        1.0,
        1.0,
        1.0,
        None,
        9,
        "Tiny external set; contains a duplicate recording"
    )

    # External results from Day 51.
    add_result(
        rows,
        "V2 - External, trimmed",
        0.8889,
        0.8333,
        0.9091,
        1.0,
        9,
        "Tiny external set; contains a duplicate recording"
    )

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT, index=False)

    print("Final evaluation report:")
    print(report.to_string(index=False))
    print("\nSaved to:", OUTPUT)
    print("\nImportant:")
    print("- External results use only nine entries.")
    print("- One WAV/MPEG pair is a duplicate recording.")
    print("- These results do not guarantee real-world accuracy.")
    print("- V1 was selected after comparing these results.")
    print("- Treat the held-out metrics as model-selection evidence,")
    print("  not as a completely untouched final evaluation.")


if __name__ == "__main__":
    main()