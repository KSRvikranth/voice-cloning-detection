from fastapi import HTTPException
from pathlib import Path
import shutil
import uuid
import tempfile 
import subprocess
import os
import whisper
import csv
from scripts.analysis_history import save_analysis
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form
)
from fastapi import Request 
from fastapi.responses import FileResponse
from scripts.voice_inference import predict_voice
from scripts.social_engineering_detector import analyze_text
from scripts.risk_scoring import calculate_risk
from scripts.explainable_warning import create_explanation
from scripts.trusted_contacts import process_trusted_alert
from scripts.analysis_report_store import (
    save_full_report,
    list_full_reports,
    get_full_report,
)
import csv
import io
from fastapi.responses import StreamingResponse
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  

FFMPEG_PATH = (
    r"C:\Users\sagas\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe"
)


def convert_to_wav(input_path, output_path):
    command = [
        FFMPEG_PATH,
        "-nostdin",
        "-y",
        "-v", "error",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise ValueError(
            "Audio conversion failed. "
            "The file may be corrupted or use an unsupported codec."
        )

    if not Path(output_path).exists():
        raise ValueError("Converted WAV file was not created.")
TEMPLATE_DIR = Path(__file__).parent / "templates"
# -----------------------------
# FastAPI application
# -----------------------------

app = FastAPI(
    title="Voice Integrity Guard",
    description=(
        "Voice cloning detection and "
        "social-engineering risk analysis"
    ),
    version="1.0.0"
)


# -----------------------------
# Configuration
# -----------------------------

UPLOAD_DIR = Path("uploads")

WHISPER_MODEL_NAME = "base"

whisper_model = None


# -----------------------------
# Load Whisper only when needed
# -----------------------------

def get_whisper_model():

    global whisper_model

    if whisper_model is None:

        print("Loading Whisper model...")

        whisper_model = whisper.load_model(
            WHISPER_MODEL_NAME
        )

    return whisper_model


# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
async def home():
    return FileResponse(TEMPLATE_DIR / "index.html")


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
def health_check():

    return {
        "status": "running"
    }

@app.get("/history")
def get_analysis_history():
    history_file = Path("results/analysis_history.csv")

    if not history_file.exists():
        return {"history": []}

    with open(history_file, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        history = list(reader)

    # Show the latest 50 analyses first
    history.reverse()
    history = history[:50]

    return {"history": history}

import csv
import io

from fastapi.responses import StreamingResponse


@app.get("/history/export")
def export_analysis_history():
    history_file = Path("results/analysis_history.csv")

    if not history_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No analysis history is available to export."
        )

    with open(
        history_file, "r", newline="", encoding="utf-8-sig"
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise HTTPException(
            status_code=404,
            detail="Analysis history is empty."
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                'attachment; filename="TrueVoice_Analysis_History.csv"'
        }
    )

@app.get("/history-reports")
def get_history_reports():
    return {"history": list_full_reports()}


@app.get("/history-reports/{report_id}")
def download_history_report_data(report_id: str):
    report = get_full_report(report_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis report not found.",
        )

    return report
# -----------------------------
# Audio analysis endpoint
# -----------------------------

@app.post("/analyze")
def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    alerts_enabled: bool = Form(False),
    minimum_risk_score: int = Form(60)
):
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            request_size = int(content_length)

            if request_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="Upload too large. Maximum request size is 25 MB."
                )

        except ValueError:
            pass
    if not 0 <= minimum_risk_score <= 100:
        raise HTTPException(
        status_code=400,
        detail="Minimum risk score must be between 0 and 100."
    )
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".mpeg",
        ".ogg",
        ".flac",
        ".m4a",
        ".aac",
        ".webm"
    }

    original_name = Path(file.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. Upload "
                "WAV, MP3, MPEG, OGG, FLAC, M4A, AAC, or WEBM."
            )
        )

    # Temporary paths for the uploaded and converted files
    input_path = None
    wav_path = None

    try:
        # Save the uploaded file using a unique temporary name
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=None
        ) as temp_input:
            input_path = Path(temp_input.name)

            shutil.copyfileobj(
                file.file,
                temp_input
            )

        # Convert every accepted format to the WAV format
        # expected by the CNN and Whisper pipeline.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            dir=None
        ) as temp_wav:
            wav_path = Path(temp_wav.name)

        print("\nConverting uploaded audio to WAV...")

        convert_to_wav(
            input_path,
            wav_path
        )

        print("Running CNN voice analysis...")

        voice_result = predict_voice(
            wav_path
        )

        print("Transcribing audio...")

        model = get_whisper_model()

        transcription_result = model.transcribe(
            str(wav_path),
            fp16=False
        )

        transcript = transcription_result[
            "text"
        ].strip()

        print("Analyzing transcript...")

        text_result = analyze_text(
            transcript
        )

        print("Calculating risk...")

        risk_result = calculate_risk(
            voice_result,
            text_result
        )

        print("Creating explanation...")

        explanation_result = create_explanation(
            risk_result
        )
        alert_result = process_trusted_alert(
            risk_result,
            alerts_enabled=alerts_enabled,
            minimum_risk_score=minimum_risk_score
        )
        # Do not trigger alerts automatically.
        # Alert opt-in will be connected to the dashboard separately.

        combined_result = {
            "audio_file": original_name,
            "voice_analysis": voice_result,
            "transcription": transcript,
            "text_analysis": text_result,
            "risk_analysis": risk_result,
            "explanation": explanation_result,
            "alert": alert_result   
        }
        save_analysis(combined_result)
        print("Analysis completed.")

        report_id = save_full_report(combined_result)
        combined_result["report_id"] = report_id

        save_analysis(combined_result)

        return combined_result
    except HTTPException:
        raise

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=422,
            detail="Audio conversion took too long."
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        print("Analysis error:", str(error))

        raise HTTPException(
            status_code=500,
            detail=(
                "Audio analysis failed. "
                "Check the server terminal."
            )
        )

    finally:
        file.file.close()

        # Delete the original upload
        if input_path is not None and input_path.exists():
            input_path.unlink()

        # Delete the converted WAV
        if wav_path is not None and wav_path.exists():
            wav_path.unlink()
# -----------------------------
# Run using:
# uvicorn app:app --reload
# -----------------------------