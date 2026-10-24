
import pandas as pd
from pathlib import Path

# Results already reported from your FoR test split
results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9723905723905724,
        "ROC_AUC": 0.9959240873696737,
        "Fake_Precision": 0.96,
        "Fake_Recall": 0.98,
        "Fake_F1": 0.97,
        "Test_Samples": 10395
    },
    {
        "Model": "CNN",
        "Accuracy": 0.9977873977873978,
        "ROC_AUC": None,
        "Fake_Precision": None,
        "Fake_Recall": None,
        "Fake_F1": None,
        "Test_Samples": 10395
    }
]

df = pd.DataFrame(results)

# Create results folder if it doesn't exist
Path("results").mkdir(exist_ok=True)

# Save comparison report
output_file = "results/model_comparison.csv"
df.to_csv(output_file, index=False)

print("\n===== MODEL COMPARISON =====")
print(df.to_string(index=False))

print(f"\nComparison saved to: {output_file}")
print("\nNote: CNN ROC-AUC and class-specific metrics")
print("were not included because they were not reported.")