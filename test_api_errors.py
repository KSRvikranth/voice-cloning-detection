
import requests

API_URL = "http://127.0.0.1:8000/analyze"


def check_response(name, response, expected_status):
    print("\nTest:", name)
    print("HTTP status:", response.status_code)
    print("Response:", response.text[:300])

    if response.status_code == expected_status:
        print("Result: PASS")
    else:
        print("Result: FAIL")
        print("Expected HTTP status:", expected_status)


print("DAY 74: API ERROR TESTS")

try:
    # Test 1: No file is uploaded.
    response = requests.post(API_URL, timeout=30)
    check_response("Missing upload", response, 422)

    # Test 2: Upload an unsupported file type.
    response = requests.post(
        API_URL,
        files={
            "file": (
                "notes.txt",
                b"This is not an audio file.",
                "text/plain"
            )
        },
        timeout=30
    )
    check_response("Unsupported file", response, 400)

    # Test 3: Upload invalid audio content with a WAV filename.
    response = requests.post(
        API_URL,
        files={
            "file": (
                "broken.wav",
                b"This is not valid WAV audio.",
                "audio/wav"
            )
        },
        timeout=60
    )
    check_response("Corrupted audio", response, 422)

except requests.exceptions.RequestException as error:
    print("\nCould not contact the API:", error)