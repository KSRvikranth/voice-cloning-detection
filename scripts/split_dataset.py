from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("results/features.csv")

TRAIN_FILE = Path("results/train.csv")
VALIDATION_FILE = Path("results/validation.csv")
TEST_FILE = Path("results/test.csv")


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Total samples:", len(df))

    # First split:
    # 70% training
    # 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Split temporary 50/50:
    # 15% validation
    # 15% testing
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print()
    print("=" * 50)
    print("DATASET SPLIT COMPLETE")
    print("=" * 50)

    print("Training:", len(train_df))
    print("Validation:", len(validation_df))
    print("Testing:", len(test_df))

    print()
    print("Files created:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()