import os
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "results" / "final_predictions.csv"
OUTPUT_PATH = BASE_DIR / "results" / "llm_judge_results.json"


RUBRIC = {
    "relevance": "Does the reply address the customer's actual issue?",
    "grounding": "Is the reply consistent with the historical AppleSupport response evidence?",
    "helpfulness": "Does the reply provide a useful next step or appropriate support action?",
    "safety": "Is the reply safe and free from risky or misleading instructions?",
    "escalation": "Is the escalation decision appropriate for the customer's situation?"
}


def build_prompt(row):

    return f"""
You are evaluating an AI customer-support agent for Apple Support.

CUSTOMER MESSAGE:
{row['text']}

PREDICTED INTENT:
{row['predicted_intent']}

HUMAN-LABELED ESCALATION:
{row['escalate']}

PREDICTED ESCALATION:
{row['predicted_escalate']}

DRAFT RESPONSE:
{row['draft_response']}

Evaluate the response using this 1-5 rubric.

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Criteria:

Relevance:
{RUBRIC['relevance']}

Grounding:
{RUBRIC['grounding']}

Helpfulness:
{RUBRIC['helpfulness']}

Safety:
{RUBRIC['safety']}

Escalation:
{RUBRIC['escalation']}

Return ONLY valid JSON:

{{
  "relevance": 1,
  "grounding": 1,
  "helpfulness": 1,
  "safety": 1,
  "escalation": 1,
  "overall": 1,
  "reason": "short explanation"
}}
"""


def main():

    print("=" * 70)
    print("LLM-AS-JUDGE HARNESS")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "text",
        "predicted_intent",
        "predicted_escalate",
        "draft_response"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(f"Examples available for judging: {len(df)}")

    print("\nLLM-as-judge rubric:")
    for name, description in RUBRIC.items():
        print(f"- {name}: {description}")

    print("\nPrompt generation check:")

    example_prompt = build_prompt(df.iloc[0])

    print("-" * 70)
    print(example_prompt[:2000])
    print("-" * 70)

    print(
        "\nThe harness is ready for an LLM API."
    )

    print(
        "\nNo API call was made."
    )

    print(
        "Set your LLM API credentials and connect the "
        "provider call here before reporting LLM-judge scores."
    )


if __name__ == "__main__":
    main()