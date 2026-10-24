
from pathlib import Path
import json
from datetime import datetime


CONFIG_FILE = Path("config/trusted_contacts.json")
ALERT_FILE = Path("results/simulated_alerts.json")


def load_config():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Trusted contacts configuration not found."
        )

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_alert(alert):

    ALERT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if ALERT_FILE.exists():

        with ALERT_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            alerts = json.load(file)

    else:
        alerts = []

    alerts.append(alert)

    with ALERT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            alerts,
            file,
            indent=4,
            ensure_ascii=False
        )


def process_trusted_alert(
    risk_result,
    alerts_enabled=False,
    minimum_risk_score=60
):

    risk_score = risk_result.get(
        "risk_score",
        0
    )

    risk_level = risk_result.get(
        "risk_level",
        "LOW"
    )

    print("\n===== TRUSTED CONTACT ALERT =====")

    print("\nCALL TAKER WARNING")
    print("Risk score:", risk_score)
    print("Risk level:", risk_level)

    print(
        "This is a potential-risk warning, "
        "not proof of fraud."
    )

    # Explicit user opt-in is required
    if not alerts_enabled:
        print("\nTrusted contact alerts are disabled.")

        return {
            "status": "DISABLED",
            "message": "Trusted contact alerts are disabled."
        }

    # Validate the selected threshold
    if (
        not isinstance(minimum_risk_score, int)
        or isinstance(minimum_risk_score, bool)
        or not 0 <= minimum_risk_score <= 100
    ):
        raise ValueError(
            "Minimum risk score must be an integer from 0 to 100."
        )

    if risk_score < minimum_risk_score:
        print("\nRisk score is below the alert threshold.")

        return {
            "status": "BELOW_THRESHOLD",
            "message": "Risk score is below the alert threshold."
        }

    # Read contacts only after explicit opt-in
    config = load_config()

    contacts = [
        contact
        for contact in config.get("contacts", [])
        if contact.get("enabled", False)
    ]

    if not contacts:
        print("\nNo enabled trusted contacts.")

        return {
            "status": "NO_CONTACTS",
            "message": "No enabled trusted contacts are configured."
        }

    alert = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "message": (
            "Potential scam risk detected. "
            "Please contact the call taker "
            "through an independent channel."
        ),
        "recipients": [
            contact["name"]
            for contact in contacts
        ],
        "status": "SIMULATED"
    }

    save_alert(alert)

    print("\nSIMULATED ALERT CREATED")

    for contact in contacts:
        print("Alert recipient:", contact["name"])

    print("\nMessage:")
    print(alert["message"])

    print("\nStatus: SIMULATED")
    print("No real message was sent.")

    return {
        "status": "SIMULATED",
        "message": alert["message"],
        "recipients": alert["recipients"]
    }


if __name__ == "__main__":

    test_risk = {
        "risk_score": 75,
        "risk_level": "HIGH"
    }

    # Test explicitly enabled simulated alert
    result = process_trusted_alert(
        test_risk,
        alerts_enabled=True,
        minimum_risk_score=60
    )

    print("\nTest result:", result)