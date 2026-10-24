
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent

V1_RESULTS = (
    ROOT / "results" / "day52_v1_trimmed_external_predictions.csv"
)
V2_RESULTS = (
    ROOT / "results" / "day51_v2_external_predictions.csv"
)
OUTPUT = ROOT / "results" / "day52_model_comparison.csv"


def load_predictions(path, model_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = {
        "file", "actual_label", "predicted_label",
        "fake_probability"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} needs columns: {sorted(required)}"
        )

    df = df.copy()
    df["file_key"] = df["file"].astype(str).apply(
        lambda name: Path(name).name
    )
    df["actual_label"] = (
        df["actual_label"].astype(str).str.lower().str.strip()
    )
    df["predicted_label"] = (
        df["predicted_label"].astype(str).str.lower().str.strip()
    )
    df = df.rename(columns={
        "actual_label": f"actual_{model_name}",
        "predicted_label": f"predicted_{model_name}",
        "fake_probability": f"fake_probability_{model_name}"
    })

    return df[
        [
            "file_key",
            f"actual_{model_name}",
            f"predicted_{model_name}",
            f"fake_probability_{model_name}"
        ]
    ]


def main():
    v1 = load_predictions(V1_RESULTS, "V1")
    v2 = load_predictions(V2_RESULTS, "V2")

    comparison = v1.merge(
        v2, on="file_key", how="inner", validate="one_to_one"
    )

    if comparison.empty:
        raise ValueError("No matching filenames between V1 and V2.")

    if not (
        comparison["actual_V1"] == comparison["actual_V2"]
    ).all():
        raise ValueError("Actual labels disagree between V1 and V2.")

    comparison["actual_label"] = comparison["actual_V1"]

    print(f"Matched audio entries: {len(comparison)}")

    for model_name in ("V1", "V2"):
        y_true = comparison["actual_label"]
        y_pred = comparison[f"predicted_{model_name}"]

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(
            "Fake recall: "
            f"{recall_score(y_true, y_pred, pos_label='fake', zero_division=0):.4f}"
        )
        print(
            "Fake F1: "
            f"{f1_score(y_true, y_pred, pos_label='fake', zero_division=0):.4f}"
        )

    comparison.to_csv(OUTPUT, index=False)
    print(f"\nSaved comparison: {OUTPUT}")


if __name__ == "__main__":
    main()