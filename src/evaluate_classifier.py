import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

MODEL_PATH = "results/intent_classifier.joblib"
GOLDEN_PATH = "evaluation/golden_set.csv"

print("=" * 70)
print("GOLDEN SET EVALUATION")
print("=" * 70)

# Load model
model = joblib.load(MODEL_PATH)

# Load human-labeled golden set
golden = pd.read_csv(GOLDEN_PATH)

# Check that labels are filled
if golden["intent"].isna().any() or (golden["intent"].astype(str).str.strip() == "").any():
    print("ERROR: Some intent labels are still empty.")
    print("Please finish labeling evaluation/golden_set.csv")
    raise SystemExit

X = golden["text"]
y_true = golden["intent"]

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"\nGolden set size: {len(golden)}")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

# Confusion matrix
labels = sorted(golden["intent"].unique())

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print("Labels:", labels)
print(cm)

# Save predictions
golden["predicted_intent"] = y_pred
golden["correct"] = golden["intent"] == golden["predicted_intent"]

golden.to_csv(
    "results/golden_predictions.csv",
    index=False
)

print("\nSaved:")
print("results/golden_predictions.csv")

print("\nIncorrect examples:", (~golden["correct"]).sum())
print("Correct examples:", golden["correct"].sum())

print("=" * 70)