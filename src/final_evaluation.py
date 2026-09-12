import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_PATH = os.path.join(BASE_DIR, "evaluation", "golden_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "intent_classifier.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "final_predictions.csv")
METRICS_PATH = os.path.join(BASE_DIR, "results", "final_metrics.json")


def load_files():
    golden = pd.read_csv(GOLDEN_PATH)
    model = joblib.load(MODEL_PATH)

    if golden["intent"].isna().any():
        raise ValueError(
            "Golden set still contains blank intent labels. "
            "Review and complete all labels first."
        )

    if golden["escalate"].isna().any():
        raise ValueError(
            "Golden set still contains blank escalation labels."
        )

    return golden, model


def main():

    print("=" * 70)
    print("FINAL APPLE SUPPORT EVALUATION")
    print("=" * 70)

    golden, model = load_files()

    texts = golden["text"].fillna("").astype(str)

    # ---------------------------------------------------------
    # MODEL PREDICTIONS
    # ---------------------------------------------------------

    predictions = model.predict(texts)

    golden["predicted_intent"] = predictions

    # ---------------------------------------------------------
    # TRIVIAL BASELINE
    # ---------------------------------------------------------

    majority_class = golden["intent"].value_counts().idxmax()

    majority_predictions = np.array(
        [majority_class] * len(golden)
    )

    majority_accuracy = accuracy_score(
        golden["intent"],
        majority_predictions
    )

    # ---------------------------------------------------------
    # SIMPLE MODEL
    # ---------------------------------------------------------

    model_accuracy = accuracy_score(
        golden["intent"],
        predictions
    )

    model_macro_f1 = f1_score(
        golden["intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    model_weighted_f1 = f1_score(
        golden["intent"],
        predictions,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)
    print(f"Golden examples: {len(golden)}")

    print("\nTRIVIAL BASELINE")
    print("-" * 70)
    print(f"Majority intent: {majority_class}")
    print(f"Accuracy: {majority_accuracy:.4f}")

    print("\nTF-IDF + LOGISTIC REGRESSION")
    print("-" * 70)
    print(f"Accuracy: {model_accuracy:.4f}")
    print(f"Macro F1: {model_macro_f1:.4f}")
    print(f"Weighted F1: {model_weighted_f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            golden["intent"],
            predictions,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    labels = sorted(golden["intent"].unique())

    cm = confusion_matrix(
        golden["intent"],
        predictions,
        labels=labels
    )

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)

    # ---------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------

    golden.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # SAVE METRICS
    # ---------------------------------------------------------

    metrics = {
        "golden_examples": int(len(golden)),
        "majority_baseline": {
            "class": majority_class,
            "accuracy": float(majority_accuracy)
        },
        "tfidf_logistic_regression": {
            "accuracy": float(model_accuracy),
            "macro_f1": float(model_macro_f1),
            "weighted_f1": float(model_weighted_f1)
        }
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFILES CREATED")
    print("-" * 70)
    print(f"Predictions: {OUTPUT_PATH}")
    print(f"Metrics:     {METRICS_PATH}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()