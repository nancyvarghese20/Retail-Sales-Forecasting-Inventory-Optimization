import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

# Feature engineering
def make_features(g):
    g = g.sort_values("date")
    for L in (1, 7, 14):
        g[f"lag_{L}"] = g["qty_sold"].shift(L)
    for W in (7, 14):
        g[f"rollmean_{W}"] = g["qty_sold"].shift(1).rolling(W).mean()
    g["dow"]  = g["date"].dt.dayofweek
    g["week"] = g["date"].dt.isocalendar().week.astype(int)
    return g

df = df.groupby(["store_id","item_id"], group_keys=False).apply(make_features).dropna()

feat_cols = ["lag_1","lag_7","lag_14","rollmean_7","rollmean_14","dow","week","price","on_promo"]
X = df[feat_cols]
y = df["qty_sold"]

# Train/test split (last 60 days = test)
split = df["date"].max() - pd.Timedelta(days=60)
Xtr, ytr = X[df["date"] <= split], y[df["date"] <= split]
Xte, yte = X[df["date"] >  split], y[df["date"] >  split]

# Train
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)

mae = mean_absolute_error(yte, model.predict(Xte))
print(f"✅ Model trained | MAE: {mae:.2f}")

# Save
joblib.dump({"model": model, "features": feat_cols}, "model/rf_model.pkl")
print("✅ Model saved to model/rf_model.pkl")