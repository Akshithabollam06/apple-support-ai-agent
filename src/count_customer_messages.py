import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/twcs.csv")

# 2. Select company tweets that reply to another tweet
company_replies = df[
    (df["inbound"] == False) &
    (df["in_response_to_tweet_id"].notna())
].copy()

# 3. Keep only the information we need
company_replies = company_replies[
    ["author_id", "in_response_to_tweet_id"]
]

# 4. Remove duplicate customer-company relationships
company_replies = company_replies.drop_duplicates()

# 5. Count customer messages for each brand
customer_counts = (
    company_replies["author_id"]
    .value_counts()
    .reset_index()
)

customer_counts.columns = [
    "brand",
    "customer_messages"
]

# 6. Display top 20 brands
top_20 = customer_counts.head(20)

print("\nTop 20 brands by customer messages:\n")
print(top_20.to_string(index=False))