

import pandas as pd
import numpy as np

df = pd.read_csv("data/retail_clean.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

def engineer_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Add lag + rolling features for one store-item group."""
    grp = grp.sort_values("date").copy()

    # ── Lag features (shift by N days) ──────────────────────────────────────
    for lag in [1, 7, 14, 21, 28]:
        grp[f"lag_{lag}"] = grp["qty_sold"].shift(lag)

    # ── Rolling mean and std (computed on already-shifted values) ───────────
    shifted = grp["qty_sold"].shift(1)          # avoid data leakage
    for window in [7, 14, 28]:
        grp[f"rollmean_{window}"] = shifted.rolling(window).mean()
        grp[f"rollstd_{window}"]  = shifted.rolling(window).std()

    # ── Expanding mean (all history up to yesterday) ─────────────────────────
    grp["expanding_mean"] = shifted.expanding().mean()

    return grp

df = (df
      .groupby(["store_id", "item_id"], group_keys=False)
      .apply(engineer_features))

# ── Label-encode categoricals ─────────────────────────────────────────────────
df["store_enc"] = df["store_id"].astype("category").cat.codes
df["item_enc"]  = df["item_id"].astype("category").cat.codes

# ── Drop rows where lags are NaN (first 28 days per group) ───────────────────
lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll")]
df = df.dropna(subset=lag_cols).reset_index(drop=True)

print(f"Feature dataset shape: {df.shape}")
print(df.columns.tolist())

df.to_csv("data/retail_features.csv", index=False)
print("Saved → data/retail_features.csv")
