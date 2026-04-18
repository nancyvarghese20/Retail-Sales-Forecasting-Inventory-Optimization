
import pandas as pd
import numpy as np

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/retail_timeseries.csv", parse_dates=["date"])
print(f"Raw shape : {df.shape}")

# ── 1. Remove stockout rows (censored demand) ────────────────────────────────
df_clean = df[df["stockout_flag"] == 0].copy()
print(f"After removing stockouts : {df_clean.shape}")

# ── 2. Ensure every SKU-store has a continuous date index ────────────────────
full_dates = pd.date_range(df_clean["date"].min(), df_clean["date"].max(), freq="D")

filled_parts = []
for (store, item), grp in df_clean.groupby(["store_id", "item_id"]):
    grp = grp.set_index("date").reindex(full_dates)
    grp["store_id"] = store
    grp["item_id"]  = item
    # forward-fill metadata columns; fill qty with 0 for missing days
    meta_cols = ["unit_cost", "ordering_cost", "holding_rate", "lead_time"]
    grp[meta_cols] = grp[meta_cols].ffill().bfill()
    grp["qty_sold"]   = grp["qty_sold"].fillna(0)
    grp["promo_flag"] = grp["promo_flag"].fillna(0).astype(int)
    grp["stockout_flag"] = 0
    filled_parts.append(grp.reset_index().rename(columns={"index": "date"}))

df_clean = pd.concat(filled_parts, ignore_index=True)
print(f"After gap-fill : {df_clean.shape}")

# ── 3. Calendar features ─────────────────────────────────────────────────────
df_clean["day_of_week"]  = df_clean["date"].dt.dayofweek     # 0=Mon
df_clean["month"]        = df_clean["date"].dt.month
df_clean["week_of_year"] = df_clean["date"].dt.isocalendar().week.astype(int)
df_clean["is_weekend"]   = (df_clean["day_of_week"] >= 5).astype(int)
df_clean["quarter"]      = df_clean["date"].dt.quarter
df_clean["year"]         = df_clean["date"].dt.year

# ── 4. Save ──────────────────────────────────────────────────────────────────
df_clean.to_csv("data/retail_clean.csv", index=False)
print("Saved → data/retail_clean.csv")
print(df_clean.dtypes)
