import json
import uuid
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_FOLDER = PROJECT_ROOT / "results" / "analysis_reports"


def save_full_report(analysis):
    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

    report_id = uuid.uuid4().hex

    report_data = {
        "report_id": report_id,
        "saved_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "analysis": analysis,
    }

    report_path = REPORTS_FOLDER / f"{report_id}.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return report_id


def list_full_reports():
    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
    reports = []

    for report_path in REPORTS_FOLDER.glob("*.json"):
        try:
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)

            analysis = report_data.get("analysis", {})
            voice = analysis.get("voice_analysis", {})
            risk = analysis.get("risk_analysis", {})
            explanation = analysis.get("explanation", {})
            text_analysis = analysis.get("text_analysis", {})

            signals = (
                explanation.get("detected_signals")
                or risk.get("signals")
                or text_analysis.get("detected_categories", [])
            )

            reports.append({
                "report_id": report_data.get("report_id", report_path.stem),
                "timestamp": report_data.get("saved_at", ""),
                "audio_file": Path(
                    analysis.get("audio_file", "unknown")
                ).name,
                "voice_prediction": voice.get("prediction", "unknown"),
                "confidence": voice.get("confidence", ""),
                "risk_score": risk.get("risk_score", ""),
                "risk_level": risk.get("risk_level", "unknown"),
                "detected_signals": "; ".join(
                    str(signal) for signal in signals
                ),
            })

        except (OSError, json.JSONDecodeError):
            continue

    reports.sort(
        key=lambda item: item.get("timestamp", ""),
        reverse=True,
    )

    return reports[:50]


def get_full_report(report_id):
    # Only allow the generated UUID-style report IDs.
    if (
        not isinstance(report_id, str)
        or len(report_id) != 32
        or any(char not in "0123456789abcdef" for char in report_id.lower())
    ):
        return None

    report_path = REPORTS_FOLDER / f"{report_id}.json"

    if not report_path.is_file():
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None