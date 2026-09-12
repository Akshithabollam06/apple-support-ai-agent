import pandas as pd
import re
from collections import Counter

TOP_BRAND = "AppleSupport"

# ---------------------------------------------------------
# We will process the CSV in chunks instead of loading
# the entire 2.8M-row dataset at once.
# ---------------------------------------------------------

chunk_size = 100_000

apple_customer_ids = set()

print("Finding AppleSupport customer messages...")

# ---------------------------------------------------------
# STEP 1: Find customer tweet IDs answered by AppleSupport
# ---------------------------------------------------------

for chunk in pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "in_response_to_tweet_id"
    ],
    dtype={
        "tweet_id": "Int64",
        "author_id": "string",
        "inbound": "boolean",
        "in_response_to_tweet_id": "string"
    },
    chunksize=chunk_size
):

    apple_responses = chunk[
        (chunk["inbound"] == False) &
        (chunk["author_id"] == TOP_BRAND)
    ]

    parent_ids = pd.to_numeric(
        apple_responses["in_response_to_tweet_id"],
        errors="coerce"
    ).dropna()

    apple_customer_ids.update(
        parent_ids.astype("int64").tolist()
    )

print(
    "Apple customer messages found:",
    len(apple_customer_ids)
)


# ---------------------------------------------------------
# STEP 2: Read customer text in chunks
# ---------------------------------------------------------

word_counts = Counter()

print("\nAnalyzing AppleSupport messages...")

stop_words = {
    "the", "and", "for", "that", "this", "with",
    "you", "are", "was", "have", "has", "had",
    "but", "not", "what", "why", "how", "when",
    "where", "who", "can", "could", "would",
    "should", "from", "about", "just", "they",
    "their", "there", "here", "your", "our",
    "out", "all", "get", "got", "its", "it's",
    "into", "been", "still", "than", "then",
    "will", "now", "too", "very", "also",
    "like", "some", "more", "one", "does",
    "did", "dont", "don't", "im", "i'm",
    "ive", "i've", "my", "me", "we", "us",
    "on", "in", "to", "of", "a", "an",
    "is", "it", "be", "or", "if", "at", "as",
    "do"
}


for chunk in pd.read_csv(
    "data/twcs.csv",
    usecols=["tweet_id", "text"],
    dtype={
        "tweet_id": "Int64",
        "text": "string"
    },
    chunksize=chunk_size
):

    # Keep only Apple customer messages
    apple_messages = chunk[
        chunk["tweet_id"].isin(apple_customer_ids)
    ]

    for text in apple_messages["text"].dropna():

        words = re.findall(
            r"\b[a-z]{3,}\b",
            str(text).lower()
        )

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        word_counts.update(words)


# ---------------------------------------------------------
# STEP 3: Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP 50 TERMS IN APPLESUPPORT CUSTOMER MESSAGES")
print("=" * 60)

for word, count in word_counts.most_common(50):
    print(f"{word:25} {count}")