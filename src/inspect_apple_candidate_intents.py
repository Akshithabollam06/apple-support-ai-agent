import csv

FILE = "data/twcs.csv"

print("=" * 70)
print("STEP 1: Finding customer tweets answered by AppleSupport")
print("=" * 70)

customer_ids = set()

with open(
    FILE,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        author_id = row["author_id"]
        inbound = row["inbound"]
        parent_id = row["in_response_to_tweet_id"]

        # AppleSupport's company reply
        if (
            author_id == "AppleSupport"
            and inbound == "False"
            and parent_id
            and parent_id != "nan"
        ):

            # response may contain multiple IDs
            for tweet_id in parent_id.split(","):

                tweet_id = tweet_id.strip()

                if tweet_id:
                    customer_ids.add(tweet_id)


print(
    "\nUnique customer tweets answered by AppleSupport:",
    len(customer_ids)
)


# ==========================================================
# STEP 2: Retrieve the actual customer messages
# ==========================================================

print("\n" + "=" * 70)
print("STEP 2: Loading actual customer messages")
print("=" * 70)

customer_messages = []

with open(
    FILE,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        tweet_id = row["tweet_id"]

        if tweet_id in customer_ids:

            customer_messages.append(
                row["text"]
            )


print(
    "\nCustomer messages loaded:",
    len(customer_messages)
)


# ==========================================================
# STEP 3: Show random examples
# ==========================================================

print("\n" + "=" * 70)
print("SAMPLE APPLESUPPORT CUSTOMER MESSAGES")
print("=" * 70)

# Deterministic sampling without pandas
import random

random.seed(42)

sample_size = min(100, len(customer_messages))

samples = random.sample(
    customer_messages,
    sample_size
)

for i, message in enumerate(samples, start=1):

    print(f"\n{i}. {message}")