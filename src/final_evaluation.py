import os
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
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_PATH = os.path.join(BASE_DIR, "evaluation", "golden_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "intent_classifier.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "results", "retriever_vectorizer.joblib")
MATRIX_PATH = os.path.join(BASE_DIR, "results", "retriever_matrix.joblib")
PAIRS_PATH = os.path.join(BASE_DIR, "results", "historical_pairs.pkl")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "final_predictions.csv")
METRICS_PATH = os.path.join(BASE_DIR, "results", "final_metrics.json")


def load_files():
    golden = pd.read_csv(GOLDEN_PATH)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    pairs = pd.read_pickle(PAIRS_PATH)

    if golden["intent"].isna().any():
        raise ValueError("Golden set contains blank intent labels.")

    if golden["escalate"].isna().any():
        raise ValueError("Golden set contains blank escalation labels.")

    return golden, model, vectorizer, matrix, pairs


def decide_escalation(message, intent):
    text = str(message).lower()

    safety_terms = [
        "dangerous", "fire", "smoke", "burning",
        "exploded", "overheating", "unsafe"
    ]

    financial_terms = [
        "charged", "charge", "refund", "payment",
        "billing", "money", "credit card"
    ]

    account_terms = [
        "locked", "verification", "verify",
        "cannot sign in", "can't sign in", "password"
    ]

    hardware_terms = [
        "broken", "cracked", "damaged",
        "won't turn on", "wont turn on"
    ]

    if any(term in text for term in safety_terms):
        return True, "safety risk"

    if intent == "app_store_purchase" and any(
        term in text for term in financial_terms
    ):
        return True, "financial dispute"

    if intent == "apple_id_account" and any(
        term in text for term in account_terms
    ):
        return True, "account verification required"

    if intent == "hardware_accessories" and any(
        term in text for term in hardware_terms
    ):
        return True, "hardware repair"

    return False, "none"


def retrieve_response(message, vectorizer, matrix, pairs):
    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    best_index = int(np.argmax(similarities))

    response_column = (
        "response_text"
        if "response_text" in pairs.columns
        else "response"
    )

    response = str(pairs.iloc[best_index][response_column])

    return response, float(similarities[best_index])


def main():

    print("=" * 70)
    print("FINAL APPLE SUPPORT EVALUATION")
    print("=" * 70)

    golden, model, vectorizer, matrix, pairs = load_files()

    texts = golden["text"].fillna("").astype(str)

    # ---------------------------------------------------------
    # INTENT PREDICTIONS
    # ---------------------------------------------------------

    predictions = model.predict(texts)

    golden["predicted_intent"] = predictions

    # ---------------------------------------------------------
    # RESPONSE + ESCALATION
    # ---------------------------------------------------------

    predicted_escalations = []
    escalation_reasons = []
    draft_responses = []
    retrieval_scores = []

    for message, intent in zip(texts, predictions):

        response, similarity = retrieve_response(
            message,
            vectorizer,
            matrix,
            pairs
        )

        escalate, reason = decide_escalation(
            message,
            intent
        )

        draft = (
            "Thanks for reaching out. "
            "Based on similar Apple Support cases, "
            "a useful next step is: "
            + response
        )

        predicted_escalations.append(escalate)
        escalation_reasons.append(reason)
        draft_responses.append(draft)
        retrieval_scores.append(similarity)

    golden["predicted_escalate"] = predicted_escalations
    golden["predicted_escalation_reason"] = escalation_reasons
    golden["draft_response"] = draft_responses
    golden["retrieval_similarity"] = retrieval_scores

    # ---------------------------------------------------------
    # CLASSIFICATION METRICS
    # ---------------------------------------------------------

    majority_class = golden["intent"].value_counts().idxmax()

    majority_predictions = np.array(
        [majority_class] * len(golden)
    )

    majority_accuracy = accuracy_score(
        golden["intent"],
        majority_predictions
    )

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
    # ESCALATION METRIC
    # ---------------------------------------------------------

    actual_escalate = (
        golden["escalate"]
        .astype(str)
        .str.lower()
        .isin(["yes", "true", "1"])
    )

    predicted_escalate = pd.Series(
        predicted_escalations
    )

    escalation_accuracy = accuracy_score(
        actual_escalate,
        predicted_escalate
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

    print("\nESCALATION")
    print("-" * 70)
    print(f"Escalation accuracy: {escalation_accuracy:.4f}")

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
        },
        "escalation": {
            "accuracy": float(escalation_accuracy)
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