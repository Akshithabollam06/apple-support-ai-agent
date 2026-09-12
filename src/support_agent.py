import pandas as pd
import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

CLASSIFIER_PATH = "results/intent_classifier.joblib"
VECTORIZER_PATH = "results/retriever_vectorizer.joblib"
MATRIX_PATH = "results/retriever_matrix.joblib"
PAIRS_PATH = "results/historical_pairs.pkl"


classifier = joblib.load(CLASSIFIER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
matrix = joblib.load(MATRIX_PATH)
pairs = pd.read_pickle(PAIRS_PATH)


# ---------------------------------------------------------
# Retrieve similar historical cases
# ---------------------------------------------------------

def retrieve_cases(message, top_k=3):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    results = []

    for idx in top_indices:

        results.append({
            "customer_message": pairs.iloc[idx]["customer_text"],
            "historical_response": pairs.iloc[idx]["response_text"],
            "similarity": float(similarities[idx])
        })

    return results


# ---------------------------------------------------------
# Escalation rules
# ---------------------------------------------------------

def decide_escalation(message, intent):

    text = message.lower()

    # Safety-related problems
    safety_terms = [
        "dangerous",
        "fire",
        "smoke",
        "burning",
        "exploded",
        "overheating",
        "unsafe"
    ]

    if any(term in text for term in safety_terms):
        return True, "safety risk"

    # Financial issues
    financial_terms = [
        "charged",
        "charge",
        "refund",
        "payment",
        "billing",
        "money",
        "credit card"
    ]

    if intent == "app_store_purchase" and any(
        term in text for term in financial_terms
    ):
        return True, "financial dispute"

    # Account verification
    account_terms = [
        "locked",
        "verification",
        "verify",
        "cannot sign in",
        "can't sign in",
        "password"
    ]

    if intent == "apple_id_account" and any(
        term in text for term in account_terms
    ):
        return True, "account verification required"

    # Physical hardware problems
    hardware_terms = [
        "broken",
        "cracked",
        "damaged",
        "won't turn on",
        "wont turn on"
    ]

    if intent == "hardware_accessories" and any(
        term in text for term in hardware_terms
    ):
        return True, "hardware repair"

    # Otherwise auto-handle
    return False, "none"


# ---------------------------------------------------------
# Response drafting
# ---------------------------------------------------------

def draft_response(message, intent, retrieved_cases):

    if not retrieved_cases:
        return (
            "Thanks for contacting Apple Support. "
            "Please share a few more details about the issue "
            "so we can help troubleshoot it."
        )

    best = retrieved_cases[0]

    historical_response = str(
        best["historical_response"]
    ).strip()

    # Use historical support language as grounding.
    response = (
        f"Thanks for reaching out. Based on similar "
        f"Apple Support cases, a useful next step is: "
        f"{historical_response}"
    )

    return response


# ---------------------------------------------------------
# Main agent
# ---------------------------------------------------------

def support_agent(message):

    # Intent
    intent = classifier.predict([message])[0]

    # Historical cases
    cases = retrieve_cases(
        message,
        top_k=3
    )

    # Escalation
    escalate, reason = decide_escalation(
        message,
        intent
    )

    # Draft response
    response = draft_response(
        message,
        intent,
        cases
    )

    return {
        "message": message,
        "intent": intent,
        "response": response,
        "escalate": escalate,
        "escalation_reason": reason,
        "retrieved_cases": cases
    }


# ---------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 70)

    while True:

        message = input(
            "\nCustomer message "
            "(type 'exit' to stop): "
        )

        if message.lower() == "exit":
            break

        result = support_agent(message)

        print("\n" + "-" * 70)
        print("INTENT:")
        print(result["intent"])

        print("\nESCALATE:")
        print(result["escalate"])

        print("REASON:")
        print(result["escalation_reason"])

        print("\nDRAFT RESPONSE:")
        print(result["response"])

        print("\nHISTORICAL CASES:")

        for i, case in enumerate(
            result["retrieved_cases"],
            start=1
        ):

            print(f"\nCase {i}")
            print(
                f"Similarity: "
                f"{case['similarity']:.3f}"
            )

            print(
                "Customer:",
                case["customer_message"]
            )

            print(
                "AppleSupport:",
                case["historical_response"]
            )