import pandas as pd

df = pd.read_csv("data/twcs.csv")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nInbound values:")
print(df["inbound"].value_counts())

print("\nData types:")
print(df.dtypes)