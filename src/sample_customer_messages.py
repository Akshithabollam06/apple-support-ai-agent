import pandas as pd

# Load the dataset
df = pd.read_csv("data/twcs.csv")

# Candidate brands we want to inspect
candidate_brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "TMobileHelp"
]

# Keep only customer messages
customer_messages = df[df["inbound"] == True].copy()

print("\nSAMPLE CUSTOMER MESSAGES\n")

for brand in candidate_brands:

    # Find company tweets for this brand
    company_tweets = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ]

    # Get customer tweet IDs that those company tweets responded to
    customer_ids = (
        company_tweets["in_response_to_tweet_id"]
        .dropna()
        .astype(float)
        .astype(int)
        .unique()
    )

    # Get those customer messages
    brand_customers = customer_messages[
        customer_messages["tweet_id"].isin(customer_ids)
    ]

    # Take a reproducible random sample
    sample_size = min(30, len(brand_customers))

    sample = brand_customers.sample(
        n=sample_size,
        random_state=42
    )

    print("\n" + "=" * 80)
    print(f"BRAND: {brand}")
    print(f"Total customer messages: {len(brand_customers)}")
    print(f"Showing {sample_size} random messages")
    print("=" * 80)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):

        print(f"\n{i}.")
        print(row["text"])