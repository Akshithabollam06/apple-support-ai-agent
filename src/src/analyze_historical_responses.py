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

# ---------------------------------------------------------
# 1. Separate customer and company tweets
# ---------------------------------------------------------

customers = df[df["inbound"] == True][
    ["tweet_id", "text"]
].rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_text"
    }
)

company = df[
    (df["inbound"] == False) &
    (df["author_id"].isin(top5))
].copy()

# ---------------------------------------------------------
# 2. Get the customer tweet that each company tweet
#    is replying to
# ---------------------------------------------------------

company["parent_tweet_id"] = pd.to_numeric(
    company["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")

company = company[
    company["parent_tweet_id"].notna()
]

# ---------------------------------------------------------
# 3. Connect company response to customer message
# ---------------------------------------------------------

pairs = company[
    [
        "author_id",
        "tweet_id",
        "text",
        "parent_tweet_id"
    ]
].rename(
    columns={
        "tweet_id": "company_tweet_id",
        "text": "company_response"
    }
)

pairs = pairs.merge(
    customers,
    left_on="parent_tweet_id",
    right_on="customer_tweet_id",
    how="inner"
)

# ---------------------------------------------------------
# 4. Basic statistics
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("HISTORICAL RESPONSE STATISTICS")
print("=" * 80)

for brand in top5:

    brand_pairs = pairs[
        pairs["author_id"] == brand
    ]

    response_count = len(brand_pairs)

    unique_customers = brand_pairs[
        "customer_tweet_id"
    ].nunique()

    avg_responses = (
        response_count / unique_customers
        if unique_customers > 0
        else 0
    )

    print(f"\nBrand: {brand}")
    print(f"Company responses: {response_count}")
    print(f"Unique customer messages answered: {unique_customers}")
    print(f"Average responses per customer message: {avg_responses:.2f}")


# ---------------------------------------------------------
# 5. Show real customer → company response examples
# ---------------------------------------------------------

for brand in top5:

    brand_pairs = pairs[
        pairs["author_id"] == brand
    ]

    sample = brand_pairs.sample(
        n=min(10, len(brand_pairs)),
        random_state=42
    )

    print("\n" + "=" * 80)
    print(f"BRAND: {brand} — HISTORICAL RESPONSE EXAMPLES")
    print("=" * 80)

    for i, (_, row) in enumerate(sample.iterrows(), 1):

        print(f"\nExample {i}")
        print("-" * 60)

        print("CUSTOMER:")
        print(row["customer_text"])

        print("\nBRAND RESPONSE:")
        print(row["company_response"])