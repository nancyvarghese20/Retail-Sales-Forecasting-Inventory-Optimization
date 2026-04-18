import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅  FIX 1 — Tell Python where the project root is
#    This file lives at:  <project_root>/src/app_streamlit.py
#    So BASE_DIR goes up one level  →  <project_root>
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅  FIX 2 — Add project root to Python path so 'src' is importable
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Now the import will work on Streamlit Cloud
from src.train_forecast_inventory import (
    load_and_clean_data,
    build_feature_matrix,
    inventory_policy,
    compute_residual_std,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED HELPERS  (✅ FIX 3 — use BASE_DIR for all file paths)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data
def load_data_and_features():
    csv_path = os.path.join(BASE_DIR, "data", "retail_timeseries.csv")
    df = load_and_clean_data(csv_path)
    df_fe, X, y, groups, feature_cols = build_feature_matrix(df)
    return df, df_fe, X, y, groups, feature_cols


@st.cache_resource
def load_trained_model():
    artifact_path = os.path.join(BASE_DIR, "outputs", "model", "retail_forecast_model.pkl")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run `python src/train_forecast_inventory.py` locally first, "
            "then push the outputs/model/ folder to GitHub."
        )
    artifact = joblib.load(artifact_path)
    return artifact["model"], artifact["features"]


def main():
    st.set_page_config(
        page_title="Retail Sales Forecast & Inventory Optimizer",
        layout="wide",
    )

    st.title("📈 Retail Sales Forecasting & Inventory Optimization")
    st.write(
        "Interactively explore **store-item sales**, see model behaviour, "
        "and get a simple **inventory recommendation**."
    )

    with st.spinner("Loading data and model..."):
        df, df_fe, X, y, groups, feature_cols = load_data_and_features()
        model, model_features = load_trained_model()

    st.sidebar.header("🔧 Controls")

    stores   = sorted(df["store_id"].unique())
    store_id = st.sidebar.selectbox("Select Store", stores)

    items   = sorted(df[df["store_id"] == store_id]["item_id"].unique())
    item_id = st.sidebar.selectbox("Select Item", items)

    on_hand        = st.sidebar.number_input("On-hand Inventory", min_value=0, value=120, step=10)
    lead_time_days = st.sidebar.slider("Lead Time (days)", min_value=1, max_value=30, value=7)
    service_level  = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Analysis")

    sku_mask = (df_fe["store_id"] == store_id) & (df_fe["item_id"] == item_id)
    sku_df   = df_fe[sku_mask].sort_values("date")

    if sku_df.empty:
        st.warning("No data for this Store–Item combination.")
        return

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📊 Historical Sales – {store_id} / {item_id}")
        hist = sku_df[["date", "qty_sold"]].tail(90).set_index("date")
        st.line_chart(hist, height=300)

    with col_right:
        st.subheader("ℹ️ SKU Snapshot")
        st.metric("Days of history",            value=len(sku_df))
        st.metric("Avg daily sales (last 30d)", value=f"{sku_df['qty_sold'].tail(30).mean():.1f}")
        st.metric("Max daily sales",            value=int(sku_df["qty_sold"].max()))

    st.markdown("---")

    if run_button:
        st.subheader("🤖 Model Behaviour & Inventory Suggestion")

        y_pred_all = model.predict(X)
        resid_std  = compute_residual_std(y, y_pred_all)

        recent_mean      = sku_df["qty_sold"].tail(30).mean()
        forecast_horizon = np.array([recent_mean] * 30)

        inv = inventory_policy(
            forecast=forecast_horizon,
            resid_std=resid_std,
            on_hand=on_hand,
            lead_time_days=lead_time_days,
            service_level=service_level,
            annual_demand=10000,
            ordering_cost=500,
            unit_cost=100,
            holding_cost_rate=0.2,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.write("### 📦 Inventory Metrics")
            st.write(f"- **Mean demand during lead time (μL)**: `{inv['mu_L']:.2f}` units")
            st.write(f"- **Safety stock**: `{inv['safety_stock']:.2f}` units")
            st.write(f"- **Reorder point**: `{inv['reorder_point']:.2f}` units")
            st.write(f"- **Economic order quantity (EOQ)**: `{inv['eoq']:.2f}` units")

        with c2:
            st.write("### ✅ Suggested Action")
            st.success(
                f"For **{store_id} / {item_id}** with **on-hand = {on_hand}** units "
                f"and **lead time = {lead_time_days} days** at **{int(service_level*100)}%** service level:\n\n"
                f"➡️ **Suggested Order Quantity:** **{inv['suggested_order_qty']:.0f} units**"
            )

        future_dates = pd.date_range(
            start=sku_df["date"].max() + pd.Timedelta(days=1),
            periods=lead_time_days,
            freq="D",
        )
        forecast_df = pd.DataFrame(
            {"date": future_dates, "forecast_qty": forecast_horizon[:lead_time_days]}
        ).set_index("date")

        st.write("### 🔮 Simple Demand Forecast (Baseline)")
        st.line_chart(forecast_df, height=250)

        st.caption(
            "Note: forecast uses recent average demand as baseline. "
            "In production, future features would be generated for the ML model."
        )


if __name__ == "__main__":
    main()
