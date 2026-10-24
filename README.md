# Voice Cloning Detection System

## 1. Project Overview

The Voice Cloning Detection System is a software-based application designed to help identify whether an audio recording may contain genuine human speech or AI-generated/cloned speech.

The system analyzes uploaded audio and provides a prediction with a risk assessment. It also examines suspicious conversation content, such as requests for OTPs, UPI payments, banking details, urgent money transfers, and impersonation.

The aim is to help users recognize potentially fraudulent or manipulated voice messages before trusting them.

## 2. Key Features

- Audio upload and analysis through a web application.
- Voice authenticity prediction using trained machine-learning models.
- Support for multiple audio formats, depending on the installed FFmpeg configuration.
- Detection of suspicious conversation patterns.
- API endpoints for submitting audio and retrieving analysis results.
- Test scripts for checking API behavior, detector performance, edge cases, and risk scoring.

*Note: The available features depend on the current application code and its configuration.*

## 3. Technologies Used

- Python
- FastAPI
- PyTorch
- Scikit-learn
- OpenAI Whisper, if required by the configured application
- FFmpeg for audio conversion
- HTML for the web interface

## 4. Project Structure

```text
Voice Cloning Detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── scripts/
├── models/
│   ├── voice_cnn.pth
│   ├── voice_cnn_multisource.pth
│   ├── voice_cnn_multisource_v2.pth
│   └── voice_detector.pkl
├── templates/
│   └── index.html
├── dataset/
├── results/
├── test_api_errors.py
├── test_api_pipeline.py
├── test_detector_regression.py
├── test_edge_cases.py
└── test_risk_scoring.py
```

Some dataset, results, and model files may be stored separately because of their size.

## 5. Requirements

Before running the project, install:

- Python 3.10 or a compatible version supported by the dependencies
- pip
- FFmpeg, if required for audio-format conversion
- The Python packages listed in `requirements.txt`
- The model files required by the current application

## 6. Installation

### Step 1: Download the repository

Download the repository as a ZIP file from GitHub and extract it, or clone it using Git.

### Step 2: Open the project folder

Open the extracted project folder in Visual Studio Code.

### Step 3: Create a virtual environment

Run the following commands in the VS Code terminal:

```powershell
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Install FFmpeg

If the application uses FFmpeg to convert uploaded audio, install FFmpeg and ensure that the `ffmpeg` command is available in the terminal.

Check with:

```powershell
ffmpeg -version
```

## 7. Model Files

The `models/` directory contains trained model files used for voice detection.

Ensure that the model files required by `app.py` and the inference scripts are present at the expected paths before starting the application.

If a model is hosted separately, download it from the project's designated model-storage location and place it in the correct `models/` directory.

## 8. Running the Application

From the project root, activate the virtual environment and install the dependencies first.

Then run:

```powershell
uvicorn app:app --reload
```

If the application starts successfully, open the local URL displayed in the terminal. For a typical FastAPI development server, this is:

`http://127.0.0.1:8000`

API documentation may be available at:

`http://127.0.0.1:8000/docs`

These addresses assume the FastAPI application object is named `app` and the default Uvicorn port is used.

## 9. Testing

The repository includes test files for different parts of the system.

Run the tests from the project root using:

```powershell
python -m pytest
```

Install `pytest` first if it is not included in `requirements.txt`.

Tests that depend on external datasets, model files, or additional services may require those resources to be available.

## 10. General Workflow

1. The user uploads an audio recording.
2. The application processes the audio into the format required by its inference pipeline.
3. The configured voice-detection model analyzes the audio.
4. The application generates an authenticity prediction.
5. Conversation analysis and risk-scoring components may identify suspicious language or social-engineering patterns.
6. The application presents the analysis and any applicable warnings to the user.

The exact processing steps and outputs depend on the implementation in `app.py` and the associated scripts.

## 11. Limitations

- Predictions are not a guarantee that a voice is genuine or AI-generated.
- Results depend on model quality, training data, recording conditions, and supported audio formats.
- A suspicious conversation score should be treated as a warning, not proof of fraud.
- The application should not be used as the sole basis for financial or security decisions.

## 12. Future Enhancements

- Improve detection accuracy across different accents, languages, and recording conditions.
- Expand testing with additional genuine and AI-generated speech datasets.
- Improve real-time audio analysis.
- Add more detailed explanations for risk scores.
- Improve the web interface and reporting features.

## 13. Academic Project

**Project:** Voice Cloning Detection System

**Purpose:** To explore the detection of AI-generated or cloned speech and identify potentially suspicious conversation patterns using software-based machine-learning techniques.
