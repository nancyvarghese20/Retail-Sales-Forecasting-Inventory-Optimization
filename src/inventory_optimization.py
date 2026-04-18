
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import norm

os.makedirs("outputs", exist_ok=True)
os.makedirs("images",  exist_ok=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
artifact = joblib.load("models/rf_model.pkl")
rf        = artifact["model"]
feat_cols = artifact["features"]
resid_info = joblib.load("models/resid_info.pkl")
resid_std  = resid_info["resid_std"]

df = pd.read_csv("data/retail_features.csv", parse_dates=["date"])

# ── Keep only last 30 days per SKU-store for a "recent" forecast proxy ────────
df = df.sort_values(["store_id", "item_id", "date"])
recent = (df
          .groupby(["store_id", "item_id"])
          .tail(30)
          .reset_index(drop=True))

X_recent = recent[feat_cols]
recent["forecast"] = np.maximum(0, rf.predict(X_recent))

SERVICE_LEVEL = 0.95
z = norm.ppf(SERVICE_LEVEL)

# ── Per-SKU-store inventory policy ────────────────────────────────────────────
records = []

for (store, item), grp in recent.groupby(["store_id", "item_id"]):
    lead_time    = int(grp["lead_time"].iloc[0])
    unit_cost    = float(grp["unit_cost"].iloc[0])
    ordering_cost = float(grp["ordering_cost"].iloc[0])
    holding_rate = float(grp["holding_rate"].iloc[0])

    # Demand during lead time
    mu_L    = grp["forecast"].head(lead_time).sum()
    sigma_L = resid_std * (lead_time ** 0.5)

    # Safety stock & ROP
    SS  = z * sigma_L
    ROP = mu_L + SS

    # Economic Order Quantity
    D_annual = grp["forecast"].mean() * 365        # annualised demand
    H        = unit_cost * holding_rate            # holding cost per unit/yr
    if H > 0 and D_annual > 0:
        EOQ = np.sqrt((2 * D_annual * ordering_cost) / H)
    else:
        EOQ = mu_L                                 # fallback

    # Assume on-hand = mean of last 7 actual days (simulation)
    on_hand = float(grp["qty_sold"].tail(7).mean() * lead_time)
    Q       = max(0.0, max(EOQ, ROP - on_hand))

    records.append({
        "store_id":        store,
        "item_id":         item,
        "lead_time_days":  lead_time,
        "mu_L (units)":    round(mu_L, 1),
        "sigma_L":         round(sigma_L, 1),
        "safety_stock":    round(SS, 1),
        "reorder_point":   round(ROP, 1),
        "EOQ":             round(EOQ, 1),
        "on_hand_est":     round(on_hand, 1),
        "order_qty":       round(Q, 1),
        "unit_cost_INR":   unit_cost,
        "order_value_INR": round(Q * unit_cost, 2),
    })

inv_df = pd.DataFrame(records)
inv_df.to_csv("outputs/inventory_recommendations.csv", index=False)
print("Inventory recommendations:\n")
print(inv_df.to_string(index=False))
print("\nSaved → outputs/inventory_recommendations.csv")

# ── Summary bar chart: Order Value by SKU ───────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
plot_df = inv_df.copy()
plot_df["sku"] = plot_df["store_id"] + " | " + plot_df["item_id"]
plot_df_sorted = plot_df.sort_values("order_value_INR", ascending=False)
ax.barh(plot_df_sorted["sku"], plot_df_sorted["order_value_INR"], color="steelblue")
ax.set_xlabel("Recommended Order Value (₹)")
ax.set_title("Recommended Purchase Order Value per SKU-Store (95% Service Level)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("images/09_order_value_by_sku.png", dpi=120)
plt.close()
print("Saved → images/09_order_value_by_sku.png")
