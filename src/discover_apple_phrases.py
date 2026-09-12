import pandas as pd
import re
from collections import Counter

TOP_BRAND = "AppleSupport"
CHUNK_SIZE = 100_000

# ---------------------------------------------------------
# STEP 1: Find customer tweets answered by AppleSupport
# ---------------------------------------------------------

apple_customer_ids = set()

print("Finding AppleSupport customer messages...")

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
    chunksize=CHUNK_SIZE
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
# STEP 2: Define stop words
# ---------------------------------------------------------

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
    "do", "please", "help", "hey", "really",
    "since", "after", "any", "only", "even",
    "going", "need", "yes", "thanks"
}


# ---------------------------------------------------------
# STEP 3: Count 2-word and 3-word phrases
# ---------------------------------------------------------

bigram_counts = Counter()
trigram_counts = Counter()

print("\nAnalyzing phrases...")

for chunk in pd.read_csv(
    "data/twcs.csv",
    usecols=["tweet_id", "text"],
    dtype={
        "tweet_id": "Int64",
        "text": "string"
    },
    chunksize=CHUNK_SIZE
):

    apple_messages = chunk[
        chunk["tweet_id"].isin(apple_customer_ids)
    ]

    for text in apple_messages["text"].dropna():

        words = re.findall(
            r"\b[a-z]{3,}\b",
            str(text).lower()
        )

        words = [
            word for word in words
            if word not in stop_words
        ]

        # 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            bigram_counts[phrase] += 1

        # 3-word phrases
        for i in range(len(words) - 2):
            phrase = (
                f"{words[i]} "
                f"{words[i + 1]} "
                f"{words[i + 2]}"
            )
            trigram_counts[phrase] += 1


# ---------------------------------------------------------
# STEP 4: Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 50 TWO-WORD PHRASES")
print("=" * 70)

for phrase, count in bigram_counts.most_common(50):
    print(f"{phrase:40} {count}")


print("\n" + "=" * 70)
print("TOP 50 THREE-WORD PHRASES")
print("=" * 70)

for phrase, count in trigram_counts.most_common(50):
    print(f"{phrase:40} {count}")