"""
Retail Sales Forecasting & Inventory Optimization System
--------------------------------------------------------
This script does EVERYTHING end-to-end:

1. Generates a synthetic retail time-series dataset if not present
2. Loads and cleans the data
3. Creates time-series features (lags, rolling stats, calendar features)
4. Trains a RandomForestRegressor model
5. Evaluates model using RMSE and R^2
6. Plots Actual vs Predicted for a sample Store-Item
7. Computes a simple inventory recommendation:
   - Safety Stock
   - Reorder Point
   - Suggested Order Quantity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import joblib
from scipy.stats import norm


# -----------------------------
# 1. DATA GENERATION (if needed)
# -----------------------------

def generate_synthetic_retail_data(
    n_stores=5,
    n_items=20,
    start_date="2022-01-01",
    end_date="2023-12-31",
    seed=42
):
    """
    Generate synthetic daily sales data for store-item combinations.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []

    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            base_demand = rng.integers(5, 50)        # base average demand
            price = rng.uniform(50, 500)             # base price

            for d in dates:
                dow = d.dayofweek  # 0=Mon, 6=Sun

                # Seasonality: weekends higher sales
                weekend_boost = 1.2 if dow >= 5 else 1.0

                # Random promotions
                on_promo = 1 if rng.random() < 0.1 else 0
                discount_pct = rng.uniform(0.1, 0.3) if on_promo else 0.0
                promo_boost = 1.3 if on_promo else 1.0

                # Trend: slowly increasing demand over time
                trend_factor = 1.0 + (d - dates[0]).days / 365 * 0.2

                # Expected demand
                lam = base_demand * weekend_boost * promo_boost * trend_factor

                # Realized demand (Poisson)
                qty_sold = rng.poisson(lam=lam)
                qty_sold = max(qty_sold, 0)

                # Random stockout: some days demand is censored to 0
                stockout_flag = 0
                if rng.random() < 0.03:
                    stockout_flag = 1
                    qty_sold = 0

                rows.append(
                    dict(
                        date=d,
                        store_id=f"S{store}",
                        item_id=f"I{item}",
                        qty_sold=qty_sold,
                        price=round(price, 2),
                        on_promo=on_promo,
                        discount_pct=round(discount_pct, 2),
                        stockout_flag=stockout_flag,
                    )
                )

    df = pd.DataFrame(rows)
    return df


def ensure_data_file(data_path="data/retail_timeseries.csv"):
    """
    If dataset is missing OR empty, generate and save it.
    """
    data_path = os.path.abspath(data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    needs_generate = True
    if os.path.exists(data_path):
        # Check if file is non-empty
        if os.path.getsize(data_path) > 0:
            needs_generate = False

    if needs_generate:
        print("Data file missing or empty. Generating synthetic dataset...")
        df = generate_synthetic_retail_data()
        df.to_csv(data_path, index=False)
        print(f"Synthetic data saved to: {data_path}")
    else:
        print(f"Data file found with content at: {data_path}")

    return data_path



# -----------------------------
# 2. DATA LOADING & CLEANING
# -----------------------------

def load_and_clean_data(csv_path):
    """
    Load CSV, handle basic quality checks, and drop stockout days.
    """
    print("\nLoading data...")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Basic sanity checks
    required_cols = {"store_id", "item_id", "date", "qty_sold"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    dup = df.duplicated(["store_id", "item_id", "date"]).sum()
    missing_qty = df["qty_sold"].isna().sum()
    neg_qty = (df["qty_sold"] < 0).sum()

    print(f"Duplicates: {dup}, Missing qty: {missing_qty}, Negative qty: {neg_qty}")

    # Keep only non-negative quantities & non-null
    df = df[df["qty_sold"].notna() & (df["qty_sold"] >= 0)].copy()

    # Remove stockout-censored rows when flag exists
    if "stockout_flag" in df.columns:
        before = len(df)
        df = df[df["stockout_flag"] == 0].copy()
        after = len(df)
        print(f"Filtered stockouts: {before - after} rows removed.")

    # Fill missing promo/discount/price columns if exist
    for c in ["on_promo", "discount_pct", "price"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

def make_features(group, lags=(1, 7, 14), windows=(7, 14, 28)):
    """
    For each store-item group, create lag and rolling window features.
    """
    group = group.sort_values("date").copy()

    # Lag features (previous days' sales)
    for L in lags:
        group[f"lag_{L}"] = group["qty_sold"].shift(L)

    # Rolling mean & std (trend and volatility)
    for W in windows:
        group[f"rollmean_{W}"] = group["qty_sold"].shift(1).rolling(W).mean()
        group[f"rollstd_{W}"] = group["qty_sold"].shift(1).rolling(W).std()

    # Calendar features
    group["dow"] = group["date"].dt.dayofweek    # 0-6
    group["weekofyear"] = group["date"].dt.isocalendar().week.astype(int)
    group["month"] = group["date"].dt.month

    return group


def build_feature_matrix(df):
    """
    Apply feature engineering per store-item and return X, y, groups.
    Ensures only numeric feature columns are used for the model.
    """
    print("\nCreating features (lags, rolling stats, calendar)...")
    df_fe = (
        df.groupby(["store_id", "item_id"], group_keys=False)
        .apply(make_features)
        .reset_index(drop=True)
    )

    # Drop rows where lag/rolling features are NaN
    df_fe = df_fe.dropna().reset_index(drop=True)

    # Columns that should NEVER go into the model
    drop_cols = ["qty_sold", "date", "store_id", "item_id", "stockout_flag"]

    # Build feature list: numeric columns only, excluding drop_cols
    feature_cols = [
        c
        for c in df_fe.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df_fe[c])
    ]

    X = df_fe[feature_cols]
    y = df_fe["qty_sold"]
    groups = df_fe["store_id"].astype(str) + "_" + df_fe["item_id"].astype(str)

    print(f"Shape after feature engineering: {X.shape}")
    print(f"Feature columns used: {feature_cols}")
    return df_fe, X, y, groups, feature_cols


# -----------------------------
# 4. MODEL TRAINING & EVALUATION
# -----------------------------

def train_model(X, y, groups):
    """
    Train a RandomForest model using GroupShuffleSplit
    so that same store-item pairs do not appear in both train & test.
    """
    print("\nSplitting train/test by groups...")
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=13)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        n_jobs=-1,
        random_state=13,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ✅ Metrics: RMSE & R^2 (manual RMSE for older sklearn)
    mse = mean_squared_error(y_test, y_pred)  # no 'squared' argument here
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R^2 : {r2:.3f}")

    return model, (X_train, X_test, y_train, y_test, y_pred)


# -----------------------------
# 5. VISUALIZATION
# -----------------------------

def plot_actual_vs_pred(df_fe, X_test, y_test, y_pred, output_path="outputs/figures"):
    """
    Plot actual vs predicted for one random store-item series.
    """
    print("\nPlotting Actual vs Predicted for a sample SKU-store...")
    os.makedirs(output_path, exist_ok=True)

    # Attach predictions to test set
    test_df = X_test.copy()
    test_df["qty_sold"] = y_test.values
    test_df["pred"] = y_pred

    # We lost store_id/item_id in features, so bring them from df_fe by index
    test_df["store_item"] = (
        df_fe.loc[X_test.index, "store_id"].astype(str)
        + "_"
        + df_fe.loc[X_test.index, "item_id"].astype(str)
    )
    test_df["date"] = df_fe.loc[X_test.index, "date"].values

    # Pick one random store-item
    sample_key = test_df["store_item"].sample(1, random_state=13).iloc[0]
    temp = test_df[test_df["store_item"] == sample_key].sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(temp["date"], temp["qty_sold"], label="Actual")
    plt.plot(temp["date"], temp["pred"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales Qty")
    plt.title(f"Actual vs Predicted Sales | {sample_key}")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_path, "sample_actual_vs_pred.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved plot to: {fig_path}")

    return sample_key, fig_path


# -----------------------------
# 6. INVENTORY POLICY FUNCTIONS
# -----------------------------

def compute_residual_std(y_true, y_pred):
    """
    Estimate residual standard deviation for uncertainty in forecast.
    """
    residuals = y_true - y_pred
    return float(np.std(residuals))


def inventory_policy(
    forecast,
    resid_std,
    on_hand,
    lead_time_days,
    service_level=0.95,
    annual_demand=10000,
    ordering_cost=500,
    unit_cost=100,
    holding_cost_rate=0.2,
):
    """
    Simple (s, Q) style inventory policy based on forecast and uncertainty.

    Returns:
        dict with safety_stock, reorder_point, eoq, order_qty
    """
    z = norm.ppf(service_level)

    # Demand during lead time
    # (sum of forecasts for next L days)
    L = int(lead_time_days)
    mu_L = float(np.sum(forecast[:L]))
    sigma_L = float(resid_std * np.sqrt(L))

    safety_stock = z * sigma_L
    reorder_point = mu_L + safety_stock

    # Economic Order Quantity (classic EOQ formula)
    H = unit_cost * holding_cost_rate
    if H <= 0:
        eoq = mu_L
    else:
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / H)

    # Suggested order quantity
    order_qty = max(0.0, max(eoq, reorder_point - on_hand))

    return dict(
        mu_L=mu_L,
        sigma_L=sigma_L,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        eoq=eoq,
        suggested_order_qty=order_qty,
    )


# -----------------------------
# 7. MAIN DRIVER
# -----------------------------

def main():
    # Step 0: ensure data exists
    data_path = ensure_data_file("data/retail_timeseries.csv")
    # Step 1: load + clean
    df = load_and_clean_data(data_path)

    # Step 2: feature engineering
    df_fe, X, y, groups, feature_cols = build_feature_matrix(df)

    # Step 3: train model & evaluate
    model, (X_train, X_test, y_train, y_test, y_pred) = train_model(X, y, groups)

    # Save model
    model_dir = os.path.abspath("outputs/model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "retail_forecast_model.pkl")
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    print(f"\nModel artifact saved to: {model_path}")

    # Step 4: visualization
    sample_key, fig_path = plot_actual_vs_pred(df_fe, X_test, y_test, y_pred)

    # Step 5: basic inventory recommendation demo
    print(f"\nComputing inventory recommendation for sample: {sample_key}")
    resid_std = compute_residual_std(y_test.values, y_pred)

    # Fake forecast horizon: assume upcoming 30 days demand similar
    # to mean prediction from test set (for demo)
    avg_pred = max(0, float(np.mean(y_pred)))
    forecast_horizon = np.array([avg_pred] * 30)

    # Assume some operational parameters
    on_hand = 120
    lead_time_days = 7

    inv = inventory_policy(
        forecast=forecast_horizon,
        resid_std=resid_std,
        on_hand=on_hand,
        lead_time_days=lead_time_days,
        service_level=0.95,
        annual_demand=10000,
        ordering_cost=500,
        unit_cost=100,
        holding_cost_rate=0.2,
    )

    print("\n=== Inventory Recommendation Demo ===")
    print(f"Sample SKU-Store: {sample_key}")
    print(f"On-hand inventory: {on_hand}")
    print(f"Lead time (days): {lead_time_days}")
    print(f"Mean demand during lead time (mu_L): {inv['mu_L']:.2f}")
    print(f"Safety stock: {inv['safety_stock']:.2f}")
    print(f"Reorder point: {inv['reorder_point']:.2f}")
    print(f"EOQ: {inv['eoq']:.2f}")
    print(f"Suggested Order Quantity: {inv['suggested_order_qty']:.2f}")
    print("====================================\n")

    # Log run summary
    os.makedirs("outputs/logs", exist_ok=True)
    with open("outputs/logs/run_log.txt", "w") as f:
        f.write("Retail Forecast & Inventory Run Completed.\n")
        f.write(f"Sample SKU-Store: {sample_key}\n")
        f.write(f"Plot: {fig_path}\n")
        f.write(str(inv))


if __name__ == "__main__":
    main()
