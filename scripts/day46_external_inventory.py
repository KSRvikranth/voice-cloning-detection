import os
import pandas as pd


# ============================================================
# EXTERNAL TEST FILES
# ============================================================

AUDIO_FILES = [

    # -------------------------
    # FAKE
    # -------------------------

    (
        "dataset/external/synthetic/voicemaker/1790060397128368606xgj5lw6p-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/179015481949335548250tu0sjb-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/1790313900119390301ihb9o7vg-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/179031391799619747fhp78u4s-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/1790313937859540443gxwgthkd-voicemaker.in-speech.mp3",
        "fake"
    ),

    (
        "dataset/external/synthetic/voicemaker/You’re the blueprint—don’t let...m4a",
        "fake"
    ),


    # -------------------------
    # REAL
    # -------------------------

    (
        "dataset/external/synthetic/voicemaker/WhatsApp Audio 2026-09-22 at 10.13.48 AM.mpeg",
        "real"
    ),

    (
        "dataset/external/synthetic/voicemaker/WhatsApp Ptt 2026-09-22 at 12.46.27 PM.ogg",
        "real"
    ),

    (
        "dataset/external/synthetic/voicemaker/WhatsApp-Audio-2026-09-22-at-10.13.48-AM.wav",
        "real"
    )
]


# ============================================================
# CREATE INVENTORY
# ============================================================

rows = []

for audio_path, label in AUDIO_FILES:

    exists = os.path.exists(audio_path)

    rows.append({
        "file": os.path.basename(audio_path),
        "path": audio_path,
        "label": label,
        "exists": exists
    })


df = pd.DataFrame(rows)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("DAY 46 - EXTERNAL DATASET INVENTORY")
print("=" * 70)

print()

for _, row in df.iterrows():

    status = "FOUND" if row["exists"] else "MISSING"

    print(
        f"{row['label'].upper():5} | "
        f"{status:7} | "
        f"{row['file']}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()

print(
    "Total files:",
    len(df)
)

print(
    "Fake files:",
    len(df[df["label"] == "fake"])
)

print(
    "Real files:",
    len(df[df["label"] == "real"])
)

print(
    "Missing files:",
    len(df[df["exists"] == False])
)


# ============================================================
# SAVE
# ============================================================

os.makedirs("results", exist_ok=True)

output_file = "results/day46_external_inventory.csv"

df.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)

print()
print("DAY 46 INVENTORY COMPLETE")