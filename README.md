# AppleSupport AI Customer Support Agent

## 1. Problem Framing

This project builds a lightweight AI customer-support agent for AppleSupport using the Customer Support on Twitter dataset.

For this project, "good" means:

1. Correctly identify the primary support intent.
2. Draft a response grounded in historically similar AppleSupport interactions.
3. Avoid automatically handling cases that require human intervention.
4. Provide an explicit reason when escalation is recommended.

I chose not to build a fully autonomous support system. The system is intentionally lightweight and focuses on intent classification, historical retrieval, response drafting, and transparent escalation rules.

## 2. Dataset and Brand Selection

Dataset: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`).

The dataset contains approximately 3 million customer-support tweets and responses across multiple brands.

I selected AppleSupport because it contains a large number of customer interactions and recurring technical-support themes.

The processed AppleSupport data contains historical customer messages and corresponding AppleSupport responses.

## 3. Intent Taxonomy

I defined 10 primary intents from recurring AppleSupport support themes:

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
Incoming customer message
          |
          v
   Intent classifier
          |
          +--------------------+
          |                    |
          v                    v
 Historical retrieval    Escalation rules
          |                    |
          v                    v
 Similar AppleSupport    Auto-handle / Human
       cases                 + reason
          |
          v
    Draft response