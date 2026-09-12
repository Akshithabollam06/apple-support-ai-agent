import pandas as pd

# Load dataset
df = pd.read_csv("data/twcs.csv")

# Top 20 brands obtained from our previous analysis
top_brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "AmericanAir",
    "Delta",
    "TMobileHelp",
    "comcastcares",
    "SouthwestAir",
    "VirginTrains",
    "Tesco",
    "Ask_Spectrum",
    "British_Airways",
    "hulu_support",
    "XboxSupport",
    "sprintcare",
    "AskPlayStation",
    "ChipotleTweets",
    "GWRHelp",
    "sainsburys"
]

# First-pass support/problem indicators
problem_words = [
    "help",
    "problem",
    "issue",
    "error",
    "wrong",
    "failed",
    "fail",
    "can't",
    "cannot",
    "unable",
    "not working",
    "refund",
    "charge",
    "charged",
    "payment",
    "order",
    "delivery",
    "late",
    "missing",
    "broken",
    "crash",
    "login",
    "account",
    "cancel",
    "complaint",
    "support",
    "why",
    "when",
    "how",
    "where"
]


def looks_like_support_message(text):
    """
    First-pass rule to identify messages
    that look like support requests/problems.
    """

    text = str(text).lower()

    # Question
    if "?" in text:
        return True

    # Problem/support keywords
    for word in problem_words:
        if word in text:
            return True

    return False


# Keep only customer/inbound messages
customer_messages = df[df["inbound"] == True].copy()

# Analyze each brand
results = []

for brand in top_brands:

    brand_messages = customer_messages[
        customer_messages["in_response_to_tweet_id"].notna()
    ]

    # We need to associate customer messages with the brand
    # using the conversation relationship.
    brand_ids = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ]["in_response_to_tweet_id"].dropna()

    brand_ids = set(brand_ids.astype(float).astype(int))

    customer_brand = customer_messages[
        customer_messages["tweet_id"].isin(brand_ids)
    ]

    total = len(customer_brand)

    support_like = customer_brand["text"].apply(
        looks_like_support_message
    ).sum()

    percentage = (
        (support_like / total) * 100
        if total > 0
        else 0
    )

    results.append({
        "brand": brand,
        "customer_messages": total,
        "support_like_messages": support_like,
        "support_like_percentage": round(percentage, 2)
    })


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by support-like messages
results_df = results_df.sort_values(
    "support_like_messages",
    ascending=False
)

print("\nTop 20 brands by support-like customer messages:\n")

print(
    results_df.to_string(index=False)
)