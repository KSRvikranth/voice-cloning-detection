
import csv
from pathlib import Path
from datetime import datetime

# Find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Store history inside the results folder.
RESULTS_FOLDER = PROJECT_ROOT / "results"
HISTORY_FILE = RESULTS_FOLDER / "analysis_history.csv"

# Columns that will be saved.
FIELDS = [
    "timestamp",
    "audio_file",
    "voice_prediction",
    "confidence",
    "risk_score",
    "risk_level",
    "detected_signals"
]


def save_analysis(analysis):
    """Save one analysis record to the history CSV."""

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    voice = analysis.get("voice_analysis", {})
    risk = analysis.get("risk_analysis", {})
    explanation = analysis.get("explanation", {})
    text = analysis.get("text_analysis", {})

    signals = (
        explanation.get("detected_signals")
        or risk.get("signals")
        or text.get("detected_categories", [])
    )

    # Avoid duplicate signal names.
    signals = sorted(set(str(signal) for signal in signals))

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "audio_file": Path(
            analysis.get("audio_file", "unknown")
        ).name,
        "voice_prediction": voice.get("prediction", "unknown"),
        "confidence": voice.get("confidence", ""),
        "risk_score": risk.get("risk_score", ""),
        "risk_level": risk.get("risk_level", "unknown"),
        "detected_signals": "; ".join(signals)
    }

    file_exists = HISTORY_FILE.exists()

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)

        if not file_exists or HISTORY_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)

    return record