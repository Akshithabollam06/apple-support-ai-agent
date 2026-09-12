import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

INPUT = "data/processed/apple_pairs.csv"
OUTPUT_DIR = "results"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BUILDING HISTORICAL RESPONSE RETRIEVER")
print("=" * 70)

# Load historical pairs
df = pd.read_csv(INPUT)

print(f"Historical pairs loaded: {len(df)}")
print("\nColumns found:")
print(df.columns.tolist())

# Find the customer and response columns
customer_col = "customer_text"

possible_response_cols = [
    "response_text",
    "text",
    "response",
    "reply",
    "reply_text"
]

response_col = None

for col in possible_response_cols:
    if col in df.columns and col != customer_col:
        response_col = col
        break

if customer_col not in df.columns:
    raise ValueError(
        f"Could not find customer_text column. Found: {df.columns.tolist()}"
    )

if response_col is None:
    raise ValueError(
        f"Could not find response column. Found: {df.columns.tolist()}"
    )

print(f"\nCustomer column: {customer_col}")
print(f"Response column: {response_col}")

# Rename to standard names
df = df.rename(columns={
    customer_col: "customer_text",
    response_col: "response_text"
})

# Remove missing values
df = df.dropna(
    subset=["customer_text", "response_text"]
).copy()

# Remove empty messages
df = df[
    (df["customer_text"].astype(str).str.strip() != "") &
    (df["response_text"].astype(str).str.strip() != "")
].copy()

print(f"\nUsable historical pairs: {len(df)}")

# Build TF-IDF index
print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

matrix = vectorizer.fit_transform(
    df["customer_text"].astype(str)
)

print(f"Matrix shape: {matrix.shape}")

# Save components
joblib.dump(
    vectorizer,
    f"{OUTPUT_DIR}/retriever_vectorizer.joblib"
)

joblib.dump(
    matrix,
    f"{OUTPUT_DIR}/retriever_matrix.joblib"
)

df.to_pickle(
    f"{OUTPUT_DIR}/historical_pairs.pkl"
)

print("\n" + "=" * 70)
print("RETRIEVER SAVED")
print("=" * 70)

print("results/retriever_vectorizer.joblib")
print("results/retriever_matrix.joblib")
print("results/historical_pairs.pkl")