import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/retail_clean.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

print(f"Input shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create lag features
for lag in [1, 7, 14, 21, 28]:
    df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(lag)

# Create rolling statistics  
for window in [7, 14, 28]:
    df[f"rollmean_{window}"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(1).rolling(window).mean()
    df[f"rollstd_{window}"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(1).rolling(window).std()

# Expanding mean
df["expanding_mean"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(1).expanding().mean()

# Label-encode categoricals
df["store_enc"] = df["store_id"].astype("category").cat.codes
df["item_enc"] = df["item_id"].astype("category").cat.codes

# Drop rows with NaN lags
lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll")]
df = df.dropna(subset=lag_cols).reset_index(drop=True)

print(f"Output shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df.to_csv("data/retail_features.csv", index=False)
print("✅ Saved → data/retail_features.csv")


import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/retail_clean.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

print(f"Input columns: {df.columns.tolist()}")
print(f"Input shape: {df.shape}")

# Create lag and rolling features
for lag in [1, 7, 14, 21, 28]:
    df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(lag)

# Create rolling statistics
for window in [7, 14, 28]:
    df[f"rollmean_{window}"] = (df.groupby(["store_id", "item_id"])["qty_sold"]
                                   .shift(1).rolling(window).mean())
    df[f"rollstd_{window}"] = (df.groupby(["store_id", "item_id"])["qty_sold"]
                                  .shift(1).rolling(window).std())

# Expanding mean (all history except today)
df["expanding_mean"] = (df.groupby(["store_id", "item_id"])["qty_sold"]
                           .shift(1).expanding().mean())

# Label-encode categoricals
df["store_enc"] = df["store_id"].astype("category").cat.codes
df["item_enc"] = df["item_id"].astype("category").cat.codes

# Drop rows where lags are NaN (first 28 days per group)
lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll")]
df = df.dropna(subset=lag_cols).reset_index(drop=True)

print(f"Feature dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Save to CSV
df.to_csv("data/retail_features.csv", index=False)
print("✅ Saved → data/retail_features.csv")