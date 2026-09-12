import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/twcs.csv",
    dtype={
        "tweet_id": "Int64",
        "author_id": "string",
        "inbound": "boolean",
        "created_at": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string"
    },
    low_memory=False
)

top5 = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "AmericanAir"
]

# Keep customer/inbound messages
customer_df = df[df["inbound"] == True].copy()

# Keep only Top 5 brands based on our brand association
# We identify the brand from company responses to customer tweets.

company_df = df[df["inbound"] == False].copy()

company_df["parent_tweet_id"] = pd.to_numeric(
    company_df["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")

# Customer tweets
customers = customer_df[
    ["tweet_id", "text"]
].rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_text"
    }
)

# Connect company response -> customer tweet
pairs = company_df[
    company_df["author_id"].isin(top5)
][
    ["author_id", "parent_tweet_id"]
].dropna()

pairs = pairs.merge(
    customers,
    left_on="parent_tweet_id",
    right_on="customer_tweet_id",
    how="inner"
)

# Remove duplicate customer messages
pairs = pairs.drop_duplicates(
    subset=["author_id", "customer_tweet_id"]
)

# Sample 30 messages from each brand
for brand in top5:

    brand_messages = pairs[
        pairs["author_id"] == brand
    ]

    sample = brand_messages.sample(
        n=min(30, len(brand_messages)),
        random_state=42
    )

    print("\n" + "=" * 80)
    print(f"BRAND: {brand}")
    print(f"Available customer messages: {len(brand_messages)}")
    print("=" * 80)

    for i, text in enumerate(sample["customer_text"], 1):
        print(f"{i}. {text}")