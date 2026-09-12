import pandas as pd
from pathlib import Path

INPUT = "data/processed/apple_customers_clean.csv"
OUTPUT = "evaluation/golden_set.csv"

Path("evaluation").mkdir(
    parents=True,
    exist_ok=True
)

# Load AppleSupport customer messages
df = pd.read_csv(INPUT)

# Reproducible random sample
golden = df.sample(
    n=200,
    random_state=42
).copy()

# Keep only the columns needed for manual annotation
golden = golden[
    ["tweet_id", "text"]
]

# Empty annotation columns
golden["intent"] = ""
golden["escalate"] = ""
golden["escalation_reason"] = ""

# Save
golden.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("GOLDEN SET CREATED")
print("=" * 70)

print("Examples:", len(golden))
print("File:", OUTPUT)

print("\nIntent labels:")
print("1. ios_update")
print("2. battery_charging")
print("3. device_performance")
print("4. apple_id_account")
print("5. app_software")
print("6. app_store_purchase")
print("7. apple_music")
print("8. icloud_backup")
print("9. hardware_accessories")
print("10. other_support")