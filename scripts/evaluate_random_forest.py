
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load the existing test dataset
test_df = pd.read_csv("results/test.csv")

# Separate features and labels
X_test = test_df.drop(columns=["file", "label"])
y_test = test_df["label"]

# Load the already-trained Random Forest model
model = joblib.load("models/voice_detector.pkl")

# Predict labels
y_pred = model.predict(X_test)

# Print evaluation results
print("\n===== RANDOM FOREST EVALUATION =====")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC if the model supports probabilities
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)

    if "fake" in classes:
        fake_index = classes.index("fake")

        y_binary = (y_test == "fake").astype(int)

        auc = roc_auc_score(
            y_binary,
            probabilities[:, fake_index]
        )

        print("\nROC-AUC:")
        print(auc)
    else:
        print("\nROC-AUC skipped: 'fake' class not found.")

else:
    print("\nROC-AUC skipped: model has no predict_proba().")