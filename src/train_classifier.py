import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_PATH = "data/processed/apple_customers_clean.csv"
GOLDEN_PATH = "evaluation/golden_set.csv"
MODEL_PATH = "results/intent_classifier.joblib"

Path("results").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("=" * 70)
print("TRAINING INTENT CLASSIFIER")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
golden = pd.read_csv(GOLDEN_PATH)

print(f"Total customer messages: {len(df)}")
print(f"Golden set size: {len(golden)}")


# ---------------------------------------------------------
# Remove golden set from training
# IMPORTANT: prevents evaluation leakage
# ---------------------------------------------------------
golden_ids = set(golden["tweet_id"].astype(str))

df["tweet_id"] = df["tweet_id"].astype(str)

train_df = df[~df["tweet_id"].isin(golden_ids)].copy()

print(f"Training messages after removing golden set: {len(train_df)}")


# ---------------------------------------------------------
# Weak labeling rules
# ---------------------------------------------------------
def assign_intent(text):
    text = str(text).lower()

    if any(x in text for x in [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "ios 19",
        "update my iphone",
        "software update",
        "new update"
    ]):
        return "ios_update"

    if any(x in text for x in [
        "battery",
        "charging",
        "charger",
        "won't charge",
        "wont charge",
        "not charging",
        "charge my phone"
    ]):
        return "battery_charging"

    if any(x in text for x in [
        "keeps crashing",
        "keep crashing",
        "keeps freezing",
        "keep freezing",
        "freezing",
        "restarting",
        "restarts",
        "slow iphone",
        "slow phone",
        "overheating"
    ]):
        return "device_performance"

    if any(x in text for x in [
        "apple id",
        "appleid",
        "icloud login",
        "can't sign in",
        "cant sign in",
        "password",
        "verification code",
        "two factor",
        "2fa"
    ]):
        return "apple_id_account"

    if any(x in text for x in [
        "app keeps",
        "app won't",
        "app wont",
        "app crashes",
        "application",
        "notification",
        "keyboard",
        "settings"
    ]):
        return "app_software"

    if any(x in text for x in [
        "app store",
        "itunes purchase",
        "refund",
        "charged for",
        "purchase",
        "download from app store"
    ]):
        return "app_store_purchase"

    if any(x in text for x in [
        "apple music",
        "itunes music",
        "music app"
    ]):
        return "apple_music"

    if any(x in text for x in [
        "icloud",
        "icloud storage",
        "icloud backup",
        "backup to icloud",
        "restore from icloud"
    ]):
        return "icloud_backup"

    if any(x in text for x in [
        "airpods",
        "earbuds",
        "headphones",
        "charging port",
        "usb port",
        "power button",
        "touch id",
        "screen is broken",
        "broken screen"
    ]):
        return "hardware_accessories"

    return "other_support"


print("\nCreating weak labels...")

train_df["intent"] = train_df["text"].apply(assign_intent)

print("\nWeak-label distribution:")
print(train_df["intent"].value_counts())


# ---------------------------------------------------------
# Remove extremely small classes
# ---------------------------------------------------------
counts = train_df["intent"].value_counts()

valid_labels = counts[counts >= 10].index

train_df = train_df[
    train_df["intent"].isin(valid_labels)
].copy()


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train_df["text"],
    train_df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=train_df["intent"]
)


print("\nTraining examples:", len(X_train))
print("Validation examples:", len(X_val))


# ---------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining model...")

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"File: {MODEL_PATH}")