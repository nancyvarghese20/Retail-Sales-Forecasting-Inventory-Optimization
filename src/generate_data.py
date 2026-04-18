
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
START_DATE  = "2022-01-01"
END_DATE    = "2023-12-31"
STORES      = ["S01", "S02", "S03"]
ITEMS       = ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D", "ITEM_E"]
UNIT_COSTS  = {"ITEM_A": 50, "ITEM_B": 120, "ITEM_C": 80, "ITEM_D": 200, "ITEM_E": 35}
HOLDING_RATE = 0.20      # 20 % of unit cost per year as holding cost
ORDERING_COST = 500      # fixed cost per purchase order (₹)
LEAD_TIME_DAYS = 7       # supplier lead time

rng = np.random.default_rng(SEED)

# ── Date range ───────────────────────────────────────────────────────────────
dates = pd.date_range(START_DATE, END_DATE, freq="D")

rows = []
for store in STORES:
    for item in ITEMS:
        base_demand = rng.integers(20, 80)          # mean daily units
        for date in dates:
            # weekly seasonality: weekends sell more
            dow_factor = 1.3 if date.dayofweek >= 5 else 1.0

            # yearly seasonality: Oct–Dec holiday spike
            month_factor = 1.5 if date.month in [10, 11, 12] else 1.0

            # random promotion on ~10 % of days
            promo = int(rng.random() < 0.10)
            promo_factor = 1.4 if promo else 1.0

            mean_qty = base_demand * dow_factor * month_factor * promo_factor
            qty_sold  = max(0, int(rng.poisson(mean_qty)))

            # stockout flag: randomly ~5 % of rows
            stockout = int(rng.random() < 0.05)
            if stockout:
                qty_sold = 0

            rows.append({
                "date":         date,
                "store_id":     store,
                "item_id":      item,
                "qty_sold":     qty_sold,
                "unit_cost":    UNIT_COSTS[item],
                "promo_flag":   promo,
                "stockout_flag": stockout,
                "lead_time":    LEAD_TIME_DAYS,
                "ordering_cost": ORDERING_COST,
                "holding_rate": HOLDING_RATE,
            })

df = pd.DataFrame(rows)
df.to_csv("data/retail_timeseries.csv", index=False)
print(f"Dataset saved → data/retail_timeseries.csv  shape={df.shape}")
print(df.head())
