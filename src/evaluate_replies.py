import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

GOLDEN_PATH = BASE_DIR / "evaluation" / "golden_set.csv"
CLASSIFIER_PATH = BASE_DIR / "results" / "intent_classifier.joblib"
RETRIEVER_VECTORIZER_PATH = BASE_DIR / "results" / "retriever_vectorizer.joblib"
RETRIEVER_MATRIX_PATH = BASE_DIR / "results" / "retriever_matrix.joblib"
HISTORICAL_PATH = BASE_DIR / "results" / "historical_pairs.pkl"

OUTPUT_PATH = BASE_DIR / "results" / "final_predictions.csv"
METRICS_PATH = BASE_DIR / "results" / "final_metrics.json"


def load_resources():

    golden = pd.read_csv(GOLDEN_PATH)

    classifier = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(RETRIEVER_VECTORIZER_PATH)
    matrix = joblib.load(RETRIEVER_MATRIX_PATH)
    historical = pd.read_pickle(HISTORICAL_PATH)

    return golden, classifier, vectorizer, matrix, historical


def retrieve_cases(message, vectorizer, matrix, historical, top_k=3):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    cases = []

    for idx in top_indices:

        row = historical.iloc[idx]

        customer_text = row.get(
            "customer_text",
            row.get("customer", "")
        )

        response_text = row.get(
            "response_text",
            row.get("response", "")
        )

        cases.append({
            "customer": str(customer_text),
            "response": str(response_text),
            "similarity": float(similarities[idx])
        })

    return cases


def decide_escalation(message, intent):

    text = message.lower()

    safety_terms = [
        "dangerous",
        "fire",
        "smoke",
        "burning",
        "exploded",
        "overheating",
        "unsafe"
    ]

    financial_terms = [
        "charged",
        "charge",
        "refund",
        "payment",
        "billing",
        "money",
        "credit card"
    ]

    account_terms = [
        "locked",
        "verification",
        "verify",
        "cannot sign in",
        "can't sign in",
        "password"
    ]

    hardware_terms = [
        "broken",
        "cracked",
        "damaged",
        "won't turn on",
        "wont turn on"
    ]

    if any(term in text for term in safety_terms):
        return True, "safety risk"

    if (
        intent == "app_store_purchase"
        and any(term in text for term in financial_terms)
    ):
        return True, "financial dispute"

    if (
        intent == "apple_id_account"
        and any(term in text for term in account_terms)
    ):
        return True, "account verification required"

    if (
        intent == "hardware_accessories"
        and any(term in text for term in hardware_terms)
    ):
        return True, "hardware repair"

    return False, "none"


def draft_response(message, cases):

    if not cases:
        return (
            "Thanks for reaching out. "
            "We're happy to help. Please provide more details "
            "about the issue so we can investigate."
        )

    best = cases[0]["response"]

    return (
        "Thanks for reaching out. "
        "Based on similar Apple Support cases, "
        "a useful next step is: "
        + best
    )


def main():

    print("=" * 70)
    print("FINAL APPLE SUPPORT AGENT EVALUATION")
    print("=" * 70)

    (
        golden,
        classifier,
        vectorizer,
        matrix,
        historical
    ) = load_resources()

    # ---------------------------------------------------------
    # CHECK GOLDEN SET
    # ---------------------------------------------------------

    if golden["intent"].isna().any():
        raise ValueError(
            "Some intent labels are blank in golden_set.csv."
        )

    if golden["escalate"].isna().any():
        raise ValueError(
            "Some escalation labels are blank in golden_set.csv."
        )

    print(f"\nGolden examples: {len(golden)}")

    # ---------------------------------------------------------
    # INTENT PREDICTIONS
    # ---------------------------------------------------------

    texts = golden["text"].fillna("").astype(str)

    predictions = classifier.predict(texts)

    golden["predicted_intent"] = predictions

    # ---------------------------------------------------------
    # GENERATE RETRIEVED CASES + REPLIES
    # ---------------------------------------------------------

    draft_replies = []
    predicted_escalations = []
    predicted_reasons = []

    print("\nGenerating grounded responses...")

    for i, message in enumerate(texts):

        cases = retrieve_cases(
            message,
            vectorizer,
            matrix,
            historical,
            top_k=3
        )

        reply = draft_response(
            message,
            cases
        )

        escalate, reason = decide_escalation(
            message,
            predictions[i]
        )

        draft_replies.append(reply)
        predicted_escalations.append(escalate)
        predicted_reasons.append(reason)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(texts)}")

    golden["draft_response"] = draft_replies
    golden["predicted_escalate"] = predicted_escalations
    golden["predicted_escalation_reason"] = predicted_reasons

    # ---------------------------------------------------------
    # INTENT METRICS
    # ---------------------------------------------------------

    majority_class = golden["intent"].value_counts().idxmax()

    majority_predictions = np.array(
        [majority_class] * len(golden)
    )

    majority_accuracy = accuracy_score(
        golden["intent"],
        majority_predictions
    )

    intent_accuracy = accuracy_score(
        golden["intent"],
        predictions
    )

    macro_f1 = f1_score(
        golden["intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        golden["intent"],
        predictions,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------
    # ESCALATION METRICS
    # ---------------------------------------------------------

    true_escalate = (
        golden["escalate"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    predicted_escalate = pd.Series(
        predicted_escalations
    )

    escalation_accuracy = accuracy_score(
        true_escalate,
        predicted_escalate
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nTRIVIAL BASELINE")
    print("-" * 70)
    print(f"Majority class: {majority_class}")
    print(f"Accuracy: {majority_accuracy:.4f}")

    print("\nINTENT CLASSIFIER")
    print("-" * 70)
    print(f"Accuracy: {intent_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nESCALATION")
    print("-" * 70)
    print(f"Accuracy: {escalation_accuracy:.4f}")

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

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )

    # ---------------------------------------------------------
    # SAVE COMPLETE PREDICTIONS
    # ---------------------------------------------------------

    golden.to_csv(
        OUTPUT_PATH,
        index=False
    )

    metrics = {
        "golden_examples": int(len(golden)),

        "majority_baseline": {
            "class": majority_class,
            "accuracy": float(majority_accuracy)
        },

        "intent_classifier": {
            "accuracy": float(intent_accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        },

        "escalation": {
            "accuracy": float(escalation_accuracy)
        }
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print("\nFILES CREATED")
    print("-" * 70)
    print(f"Predictions: {OUTPUT_PATH}")
    print(f"Metrics:     {METRICS_PATH}")

    print("\nFinal evaluation complete.")


if __name__ == "__main__":
    main()