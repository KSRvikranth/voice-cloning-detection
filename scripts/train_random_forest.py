from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


TRAIN_FILE = Path("results/train.csv")
TEST_FILE = Path("results/test.csv")

MODEL_FILE = Path("models/voice_detector.pkl")


def main():

    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    feature_columns = [
        column
        for column in train_df.columns
        if column not in ["file", "label"]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features:", len(feature_columns))

    print()
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)

    print("Accuracy:", accuracy)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()
    print("Model saved to:")
    print(MODEL_FILE.resolve())


if __name__ == "__main__":
    main()