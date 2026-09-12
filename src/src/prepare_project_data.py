import pandas as pd
import re
from pathlib import Path

FILE = "data/twcs.csv"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Find customer messages answered by AppleSupport
# ============================================================

print("Finding AppleSupport customer messages...")

customer_ids = set()

for chunk in pd.read_csv(
    FILE,
    usecols=[
        "author_id",
        "inbound",
        "in_response_to_tweet_id"
    ],
    dtype={
        "author_id": "string",
        "inbound": "boolean",
        "in_response_to_tweet_id": "string"
    },
    chunksize=100000,
    low_memory=False
):

    apple_replies = chunk[
        (chunk["author_id"] == "AppleSupport") &
        (chunk["inbound"] == False)
    ]

    for value in apple_replies["in_response_to_tweet_id"].dropna():

        for tweet_id in str(value).split(","):

            tweet_id = tweet_id.strip()

            if tweet_id:
                customer_ids.add(tweet_id)


print("Customer IDs:", len(customer_ids))


# ============================================================
# 2. Load customer messages
# ============================================================

print("Loading customer messages...")

customer_rows = []

for chunk in pd.read_csv(
    FILE,
    usecols=["tweet_id", "text"],
    dtype={
        "tweet_id": "string",
        "text": "string"
    },
    chunksize=100000,
    low_memory=False
):

    matched = chunk[
        chunk["tweet_id"].isin(customer_ids)
    ]

    if not matched.empty:
        customer_rows.append(matched)


customers = pd.concat(
    customer_rows,
    ignore_index=True
)

customers = customers.drop_duplicates(
    subset=["tweet_id"]
)

customers.to_csv(
    OUT_DIR / "apple_customers.csv",
    index=False
)

print(
    "Saved customers:",
    len(customers)
)


# ============================================================
# 3. Extract historical customer → AppleSupport responses
# ============================================================

print("Building historical response pairs...")

pairs = []

for chunk in pd.read_csv(
    FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id"
    ],
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "in_response_to_tweet_id": "string"
    },
    chunksize=100000,
    low_memory=False
):

    replies = chunk[
        (chunk["author_id"] == "AppleSupport") &
        (chunk["inbound"] == False)
    ]

    if not replies.empty:
        pairs.append(
            replies[
                [
                    "tweet_id",
                    "text",
                    "in_response_to_tweet_id"
                ]
            ]
        )


responses = pd.concat(
    pairs,
    ignore_index=True
)

responses = responses.rename(
    columns={
        "tweet_id": "response_id",
        "text": "response",
        "in_response_to_tweet_id": "customer_id"
    }
)

# One response ID may reference one customer tweet
responses["customer_id"] = (
    responses["customer_id"]
    .str.split(",")
)

responses = responses.explode(
    "customer_id"
)

responses["customer_id"] = (
    responses["customer_id"]
    .str.strip()
)

responses = responses.merge(
    customers,
    left_on="customer_id",
    right_on="tweet_id",
    how="inner"
)

responses = responses.rename(
    columns={
        "text": "customer_text"
    }
)

responses = responses[
    [
        "customer_id",
        "customer_text",
        "response"
    ]
]

responses = responses.drop_duplicates()

responses.to_csv(
    OUT_DIR / "apple_pairs.csv",
    index=False
)

print(
    "Saved historical pairs:",
    len(responses)
)


# ============================================================
# 4. Clean text
# ============================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"https?://\S+",
        " URL ",
        text
    )

    text = re.sub(
        r"@\w+",
        " USER ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


customers["clean_text"] = (
    customers["text"]
    .fillna("")
    .apply(clean_text)
)

customers.to_csv(
    OUT_DIR / "apple_customers_clean.csv",
    index=False
)

print("\nDONE.")