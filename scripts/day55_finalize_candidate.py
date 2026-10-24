
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

V1_RESULTS = (
    ROOT / "results" /
    "day52_v1_trimmed_external_predictions.csv"
)

V2_RESULTS = (
    ROOT / "results" /
    "day51_v2_external_predictions.csv"
)

V1_MODEL = ROOT / "models" / "voice_cnn_multisource.pth"
V2_MODEL = ROOT / "models" / "voice_cnn_multisource_v2.pth"

OUTPUT = ROOT / "results" / "day55_candidate_summary.csv"


def main():
    # Check that the model files exist.
    if not V1_MODEL.exists():
        print("V1 model file not found.")
        return

    if not V2_MODEL.exists():
        print("V2 model file not found.")
        return

    # Check that the evaluation files exist.
    if not V1_RESULTS.exists() or not V2_RESULTS.exists():
        print("Evaluation CSV files not found.")
        return

    # Read the existing evaluation results.
    v1 = pd.read_csv(V1_RESULTS)
    v2 = pd.read_csv(V2_RESULTS)

    # Change this column name if your CSV uses another name.
    print("V1 columns:", list(v1.columns))
    print("V2 columns:", list(v2.columns))

    # Calculate the metrics from the recorded predictions.
    def calculate_metrics(data):
        actual = data["actual_label"].astype(str).str.lower()
        predicted = data["predicted_label"].astype(str).str.lower()

        correct = (actual == predicted).sum()
        total = len(data)
        accuracy = correct / total if total else 0

        fake_actual = actual == "fake"
        fake_predicted = predicted == "fake"

        true_fake = (fake_actual & fake_predicted).sum()
        missed_fake = (fake_actual & ~fake_predicted).sum()

        fake_recall = (
            true_fake / (true_fake + missed_fake)
            if (true_fake + missed_fake) else 0
        )

        precision = (
            true_fake / fake_predicted.sum()
            if fake_predicted.sum() else 0
        )

        f1 = (
            2 * precision * fake_recall / (precision + fake_recall)
            if (precision + fake_recall) else 0
        )

        return {
            "model": "",
            "files_evaluated": total,
            "accuracy": round(accuracy, 4),
            "fake_precision": round(precision, 4),
            "fake_recall": round(fake_recall, 4),
            "fake_f1": round(f1, 4),
            "missed_fake_files": int(missed_fake),
        }

    # Calculate results separately.
    v1_metrics = calculate_metrics(v1)
    v1_metrics["model"] = "V1 + trimmed preprocessing"

    v2_metrics = calculate_metrics(v2)
    v2_metrics["model"] = "V2 + trimmed preprocessing"

    # Save a summary for the project report.
    summary = pd.DataFrame([v1_metrics, v2_metrics])
    summary.to_csv(OUTPUT, index=False)

    print("\nCandidate comparison:")
    print(summary.to_string(index=False))
    print("\nSaved summary to:", OUTPUT)

    print("\nCurrent provisional candidate:")
    print("V1 + trimmed preprocessing")
    print("Keep V2 for future experiments.")
    print("Do not claim guaranteed real-world accuracy.")


if __name__ == "__main__":
    main()