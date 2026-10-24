
import os
import requests

# Change this to the path of an audio file that already exists.
AUDIO_PATH = r"C:\Users\sagas\OneDrive\Desktop\Voice Cloning Detection\dataset\test\sample.wav"

API_URL = "http://127.0.0.1:8000/analyze"


def get_all_keys(data):
    """Collect keys from nested dictionaries."""
    keys = set()

    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key)
            keys.update(get_all_keys(value))

    elif isinstance(data, list):
        for item in data:
            keys.update(get_all_keys(item))

    return keys


print("DAY 73: COMPLETE PIPELINE TEST")

# Check whether the audio file exists.
if not os.path.isfile(AUDIO_PATH):
    print("FAIL: Audio file not found:", AUDIO_PATH)
    print("Update AUDIO_PATH with an existing audio file.")
    raise SystemExit(1)

try:
    # Upload the audio file to FastAPI.
    with open(AUDIO_PATH, "rb") as audio:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    os.path.basename(AUDIO_PATH),
                    audio
                )
            },
            timeout=300
        )

    print("HTTP status:", response.status_code)

    # Check whether the API returned a successful response.
    if response.status_code != 200:
        print("FAIL: API returned an error.")
        print(response.text)
        raise SystemExit(1)

    # Read the JSON response.
    result = response.json()

    if not isinstance(result, dict):
        print("FAIL: Expected a JSON object.")
        raise SystemExit(1)

    print("\nReturned result:")
    print(result)

    # Check whether the response contains analysis fields.
    keys = get_all_keys(result)
    print("\nResponse fields:", sorted(keys))

    expected_fields = {
        "prediction",
        "transcript",
        "risk_score",
        "risk_level"
    }

    found_fields = expected_fields.intersection(keys)

    if found_fields:
        print("\nPASS: Analysis response received.")
        print("Fields found:", sorted(found_fields))
    else:
        print("\nWARNING: Response received, but")
        print("the expected analysis fields were not found.")
        print("Check the field names in your API response.")

except requests.exceptions.RequestException as error:
    print("FAIL: Could not contact the API.")
    print("Details:", error)

except ValueError:
    print("FAIL: The API response was not valid JSON.")