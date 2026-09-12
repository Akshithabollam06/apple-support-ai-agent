# Apple Support AI Agent

A retrieval-grounded customer-support agent built from the Customer Support on Twitter dataset.

## 1. Problem

The system takes an incoming Apple customer message and:

1. Classifies the support intent.
2. Retrieves historically similar AppleSupport cases.
3. Drafts a response grounded in historical support behavior.
4. Decides whether the case should be escalated to a human.

## 2. Dataset

Dataset:
Customer Support on Twitter / thoughtvector/customer-support-on-twitter

The project focuses only on the AppleSupport brand.

The extracted AppleSupport customer-message dataset contains approximately 106K usable customer interactions.

## 3. Intent taxonomy

The system uses 10 intents:

- ios_update
- battery_charging
- device_performance
- apple_id_account
- app_software
- app_store_purchase
- apple_music
- icloud_backup
- hardware_accessories
- other_support

## 4. System architecture

Customer message
        |
        v
TF-IDF + Logistic Regression
        |
        v
Intent
        |
        +------------------+
        |                  |
        v                  v
Historical retrieval   Escalation rules
        |                  |
        v                  v
Grounded reply       Human / Auto
        |
        v
Final response

## 5. Installation

Python 3.13 was used during development.

Install dependencies:

```bash
pip install -r requirements.txt