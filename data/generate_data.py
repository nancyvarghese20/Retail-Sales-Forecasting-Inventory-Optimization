import pandas as pd
import numpy as np

np.random.seed(42)

stores = ["S1", "S2", "S3"]
items  = ["Rice", "Oil", "Sugar", "Biscuits", "Soap"]
dates  = pd.date_range("2023-01-01", "2024-12-31", freq="D")

rows = []
for store in stores:
    for item in items:
        base = np.random.randint(20, 100)
        for date in dates:
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise       = np.random.normal(0, 5)
            qty         = max(0, int(base * seasonality + noise))
            price       = round(np.random.uniform(10, 200), 2)
            on_promo    = np.random.choice([0, 1], p=[0.8, 0.2])
            rows.append([store, item, date, qty, price, on_promo])

df = pd.DataFrame(rows, columns=["store_id","item_id","date","qty_sold","price","on_promo"])
df.to_csv("data/sales_data.csv", index=False)
print("✅ Dataset created:", df.shape)