# AppleSupport AI Customer Support Agent

## 1. Problem Framing

This project builds a lightweight AI customer-support agent for AppleSupport using the Customer Support on Twitter dataset.

The agent:
1. Identifies the primary support intent.
2. Drafts a response grounded in historically similar AppleSupport interactions.
3. Decides whether to auto-handle or escalate to a human.
4. Provides an explicit escalation reason.

The system is intentionally lightweight rather than a fully autonomous support system.

## 2. Dataset and Brand Selection

Dataset: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`).

The dataset contains approximately 2.8 million tweets/replies across multiple brands.

AppleSupport was selected because it has many customer-support interactions and diverse technical-support themes.

The processed AppleSupport data contains approximately 106K customer messages with historical responses.

## 3. Intent Taxonomy

The system uses 10 primary intents:

- `ios_update`
- `battery_charging`
- `device_performance`
- `apple_id_account`
- `app_software`
- `app_store_purchase`
- `apple_music`
- `icloud_backup`
- `hardware_accessories`
- `other_support`

Each message receives one primary intent.

## 4. System Architecture

```text
Customer message
       |
       v
TF-IDF + Logistic Regression
       |
       +----------------------+
       |                      |
       v                      v
Historical retrieval     Escalation rules
       |                      |
       v                      v
Similar AppleSupport     Auto-handle /
cases                    Human + reason
       |
       v
Draft response
```

The response system retrieves similar historical AppleSupport cases using TF-IDF similarity and uses the best historical response as the basis for the draft.

## 5. Evaluation Setup

A 200-example golden set was created from AppleSupport customer messages.

Golden examples were excluded from classifier training to reduce direct evaluation leakage.

Two baselines were used:
- Trivial baseline: majority-class prediction.
- Simple ML baseline: TF-IDF + Logistic Regression.

Training labels were generated using keyword-based weak-label rules. Therefore, weak-label validation performance is not used as the headline result.

## 6. Results

| Metric | Result |
|---|---:|
| Majority baseline accuracy | 25.50% |
| Classifier accuracy | 42.00% |
| Macro F1 | 43.18% |
| Weighted F1 | 38.22% |
| Escalation accuracy | 92.00% |

The classifier improves over the majority baseline by 16.5 percentage points in accuracy.

The 42.0% golden-set accuracy is the primary classifier result.

## 7. What Is Misleading About My Headline Number?

The 97.7% validation accuracy obtained during training is misleading because the validation labels were generated using keyword-based weak-label rules.

The more meaningful result is 42.0% accuracy on the held-out golden set.

This gap shows that weak labels do not perfectly represent real customer intent.

## 8. Top Failure Modes

### 1. Over-prediction of `other_support`
Many `app_software`, `device_performance`, and `ios_update` messages are classified as `other_support`.

### 2. Mismatched historical retrieval
TF-IDF retrieval can return responses that are related to Apple Support but do not directly solve the customer's issue.

### 3. Twitter-specific response artifacts
Historical responses may contain usernames, DM instructions, and shortened URLs.

### 4. Rare intents
Rare classes have fewer evaluation examples and are harder to classify reliably.

### 5. Short context-dependent messages
Messages such as "DO SOMETHING" or "I've sent you a DM" require conversation history that the current system does not fully use.

## 9. Reply Quality Review

A preliminary model-assisted review of 20 generated replies produced:

| Criterion | Average / 5 |
|---|---:|
| Relevance | 3.65 |
| Grounding | 3.90 |
| Helpfulness | 3.50 |
| Safety | 5.00 |
| Escalation | 4.75 |
| Overall | 3.60 |

These ratings were model-assisted and are not presented as independent human evaluation.

An LLM-as-judge harness is included in `src/llm_judge.py`.

No external LLM API was available during this submission, so actual LLM-judge scores and judge-human agreement are not claimed.

## 10. Escalation Policy

The system escalates cases involving:
- Safety risks
- Financial disputes
- Account verification problems
- Hardware repair issues
- Complex unresolved problems

Otherwise, the system attempts to provide an automatically drafted response.

## 11. Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/train_classifier.py
python src/evaluate_classifier.py
python src/build_retriever.py
python src/support_agent.py
python src/final_evaluation.py
python src/evaluate_replies.py
python src/llm_judge.py
```

## 12. One-More-Week Plan

1. Replace keyword weak labels with higher-quality human labels.
2. Improve the intent taxonomy using confusion analysis.
3. Use sentence embeddings for semantic retrieval.
4. Clean historical responses before using them as drafts.
5. Add conversation context for short follow-up messages.
6. Run an actual LLM-as-judge evaluation with independent human ratings.

## 13. Limitations

The main limitations are weak-label noise, overlapping intents, class imbalance, a relatively small golden set, limited conversation context, and Twitter-specific historical response artifacts.

The system should therefore be treated as a prototype support-assistance system rather than a production-ready autonomous agent.

## 14. Repository Structure

```text
customer support/
├── data/
│   └── processed/
├── evaluation/
│   ├── golden_set.csv
│   └── human_reply_review.csv
├── results/
├── src/
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── build_retriever.py
│   ├── support_agent.py
│   ├── final_evaluation.py
│   ├── evaluate_replies.py
│   └── llm_judge.py
├── DECISION_LOG.md
├── README.md
├── requirements.txt
└── .gitignore
```
