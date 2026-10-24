
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

V1_FILE = (
    ROOT / "results" / "day52_v1_trimmed_external_predictions.csv"
)
V2_FILE = (
    ROOT / "results" / "day51_v2_external_predictions.csv"
)

OUTPUT = ROOT / "results" / "day53_false_negative_analysis.csv"


def find_false_negatives(file_path, model_name):
    df = pd.read_csv(file_path)

    required = {
        "file", "actual_label",
        "predicted_label", "fake_probability"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{file_path.name} is missing required columns."
        )

    df["actual_label"] = (
        df["actual_label"].astype(str).str.lower().str.strip()
    )
    df["predicted_label"] = (
        df["predicted_label"].astype(str).str.lower().str.strip()
    )

    # False negative: actual audio is fake, but prediction is real.
    missed = df[
        (df["actual_label"] == "fake")
        & (df["predicted_label"] == "real")
    ].copy()

    missed["model"] = model_name
    return missed[
        ["file", "actual_label", "predicted_label",
         "fake_probability", "model"]
    ]


v1_missed = find_false_negatives(V1_FILE, "V1")
v2_missed = find_false_negatives(V2_FILE, "V2")

all_missed = pd.concat(
    [v1_missed, v2_missed],
    ignore_index=True
)

if all_missed.empty:
    print("No false negatives found in either external prediction file.")
else:
    print("False negatives found:")
    print(all_missed.to_string(index=False))

all_missed.to_csv(OUTPUT, index=False)
print(f"\nSaved analysis to: {OUTPUT}")