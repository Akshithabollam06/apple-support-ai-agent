# Decision Log — Apple Support AI Agent

This log records non-obvious decisions made during the development and evaluation of the Apple Support AI agent.

## 1. Selected AppleSupport as the brand

I selected AppleSupport because it had a large number of customer-support interactions in the dataset and contained diverse technical-support issues such as iOS updates, battery problems, account issues, applications, iCloud, music, and hardware.

## 2. Used customer messages answered by AppleSupport

Instead of treating every tweet in the dataset as a support request, I identified customer messages associated with AppleSupport responses. This better represents the actual customer-support workflow.

## 3. Used historical customer-response pairs

For response grounding, customer messages were linked to the corresponding AppleSupport responses. This allows the agent to retrieve how similar issues were historically handled.

## 4. Defined a small intent taxonomy

I used 10 primary intents:

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

The taxonomy was designed around recurring themes observed in AppleSupport conversations.

## 5. Assigned one primary intent per message

Some customer messages contain multiple problems, such as an update problem combined with battery drain. I assigned one primary intent to keep the classification task deterministic and easy to evaluate.

## 6. Created a 200-example golden evaluation set

I used 200 AppleSupport customer messages as the evaluation set. This is within the required 150–250 example range.

## 7. Kept the golden examples out of classifier training

The golden-set tweet IDs were excluded from the classifier training data to reduce direct evaluation leakage.

## 8. Used weak labeling for scalable training

Rather than manually labeling more than 100,000 historical messages, keyword-based rules were used to create weak training labels. This allowed a simple classifier to be trained quickly.

## 9. Chose TF-IDF + Logistic Regression

TF-IDF with Logistic Regression was selected as the simple machine-learning baseline because it is fast, interpretable, lightweight, and appropriate for text classification.

## 10. Included a majority-class baseline

A majority-class classifier was included as the trivial baseline. It predicts the most common intent for every customer message and provides a simple reference point for measuring whether the trained classifier adds value.

## 11. Used historical retrieval for response grounding

Instead of generating unsupported answers, the agent retrieves similar historical AppleSupport cases and uses the best matching historical response as the basis for its draft reply.

## 12. Retrieved the top three similar cases

The agent retrieves three similar historical cases using TF-IDF similarity. This provides multiple pieces of historical context while keeping the system lightweight.

## 13. Added explicit escalation rules

Escalation is handled using transparent rules for situations such as safety risks, financial disputes, account verification problems, hardware repair, and complex unresolved issues.

## 14. Evaluated the complete pipeline on the golden set

The final evaluation measures the classifier and escalation behavior on the held-out 200-example golden set rather than relying on the much easier weak-label validation score.

## 15. Treated the gap between weak-label and golden performance as a finding

The classifier achieved much stronger performance against weak validation labels than against the golden set. Rather than presenting the weak-label result as the headline number, I use the golden-set result and discuss this gap as evidence that the weak labels do not perfectly represent real customer intent.

## 16. Kept the system lightweight

The implementation uses classical NLP and retrieval rather than a large end-to-end generative model. This keeps training and evaluation reproducible within the assessment's runtime constraint.

## 17. Kept original historical response artifacts visible

Historical responses may contain Twitter-specific elements such as usernames, "DM us" language, or shortened links. These artifacts were retained as evidence of historical behavior rather than silently rewriting the source data.

## 18. Added an LLM-as-judge evaluation design

Reply quality is evaluated using a structured rubric covering relevance, grounding, helpfulness, safety, and escalation appropriateness. The judge is intended to complement automated classification and escalation metrics.

## 19. Reported limitations explicitly

The main limitations are weak-label noise, overlapping intents, small golden-set size, class imbalance, and historical responses containing platform-specific artifacts.

## 20. Prioritized failure analysis over headline optimization

The goal was not simply to maximize one metric. The evaluation is designed to expose where the system fails and provide concrete directions for improving it.