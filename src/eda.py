
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("data/retail_clean.csv", parse_dates=["date"])

# ── 1. Total daily sales across all stores ───────────────────────────────────
daily = df.groupby("date")["qty_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily["date"], daily["qty_sold"], linewidth=0.8)
ax.set_title("Total Daily Sales (All Stores & Items)")
ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
plt.tight_layout()
plt.savefig("images/01_total_daily_sales.png", dpi=120)
plt.close()
print("Saved: images/01_total_daily_sales.png")

# ── 2. Sales by item ─────────────────────────────────────────────────────────
item_sales = df.groupby("item_id")["qty_sold"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
item_sales.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Total Sales by Item"); ax.set_ylabel("Units Sold")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("images/02_sales_by_item.png", dpi=120)
plt.close()
print("Saved: images/02_sales_by_item.png")

# ── 3. Sales by store ────────────────────────────────────────────────────────
store_sales = df.groupby("store_id")["qty_sold"].sum()
fig, ax = plt.subplots(figsize=(6, 4))
store_sales.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
ax.set_title("Total Sales by Store"); ax.set_ylabel("Units Sold")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("images/03_sales_by_store.png", dpi=120)
plt.close()
print("Saved: images/03_sales_by_store.png")

# ── 4. Seasonal heatmap — month × day-of-week ────────────────────────────────
pivot = df.pivot_table(values="qty_sold", index="month",
                       columns="day_of_week", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set_title("Avg Daily Sales: Month vs Day-of-Week")
ax.set_xlabel("Day of Week (0=Mon)"); ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("images/04_seasonal_heatmap.png", dpi=120)
plt.close()
print("Saved: images/04_seasonal_heatmap.png")

# ── 5. Monthly sales trend by item ──────────────────────────────────────────
df["year_month"] = df["date"].dt.to_period("M").astype(str)
monthly_item = df.groupby(["year_month", "item_id"])["qty_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 5))
for item, grp in monthly_item.groupby("item_id"):
    ax.plot(grp["year_month"], grp["qty_sold"], marker="o", label=item, markersize=3)
ax.set_title("Monthly Sales Trend by Item")
ax.set_xlabel("Month"); ax.set_ylabel("Units Sold")
plt.xticks(rotation=45, fontsize=7)
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig("images/05_monthly_trend_by_item.png", dpi=120)
plt.close()
print("Saved: images/05_monthly_trend_by_item.png")

# ── 6. Promo vs non-promo sales distribution ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
df[df["promo_flag"] == 0]["qty_sold"].hist(bins=40, alpha=0.6, label="No Promo", ax=ax)
df[df["promo_flag"] == 1]["qty_sold"].hist(bins=40, alpha=0.6, label="Promo", ax=ax, color="orange")
ax.set_title("Sales Distribution: Promo vs Non-Promo")
ax.set_xlabel("Units Sold"); ax.legend(); plt.tight_layout()
plt.savefig("images/06_promo_distribution.png", dpi=120)
plt.close()
print("Saved: images/06_promo_distribution.png")

print("\nEDA complete. All charts saved to images/")
