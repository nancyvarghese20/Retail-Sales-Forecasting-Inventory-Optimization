import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Retail Forecast Dashboard", layout="wide")
st.title("🛒 Retail Sales Forecasting & Inventory Optimization")

# ── Load data & model ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/sales_data.csv", parse_dates=["date"])

@st.cache_resource
def load_model():
    return joblib.load("model/rf_model.pkl")

df    = load_data()
art   = load_model()
model = art["model"]
feats = art["features"]

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.header("🔧 Select Options")
store = st.sidebar.selectbox("Store",  sorted(df["store_id"].unique()))
item  = st.sidebar.selectbox("Product", sorted(df["item_id"].unique()))
on_hand    = st.sidebar.number_input("Current Stock On Hand (units)", min_value=0, value=100)
lead_time  = st.sidebar.slider("Supplier Lead Time (days)", 1, 30, 7)
service_lv = st.sidebar.slider("Service Level (%)", 80, 99, 95)

# ── Filter SKU data ───────────────────────────────────────────
sku_df = df[(df["store_id"]==store) & (df["item_id"]==item)].sort_values("date").copy()

# ── Tab layout ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Sales Trend", "🔮 Forecast", "📦 Inventory Plan"])

# ══ TAB 1: Sales Trend ═══════════════════════════════════════
with tab1:
    st.subheader(f"Historical Sales — {item} @ {store}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Sales", f"{sku_df['qty_sold'].mean():.1f} units")
    col2.metric("Max Daily Sales", f"{sku_df['qty_sold'].max()} units")
    col3.metric("Zero-Sale Days",  f"{(sku_df['qty_sold']==0).sum()} days")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sku_df["date"], sku_df["qty_sold"], color="#4C9BE8", linewidth=0.8)
    ax.set_xlabel("Date"); ax.set_ylabel("Qty Sold")
    ax.set_title("Daily Sales History")
    st.pyplot(fig)

    # Weekly aggregation
    weekly = sku_df.resample("W", on="date")["qty_sold"].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(weekly.index, weekly.values, color="#F4845F", width=5)
    ax2.set_xlabel("Week"); ax2.set_ylabel("Weekly Qty")
    ax2.set_title("Weekly Sales Aggregation")
    st.pyplot(fig2)

# ══ TAB 2: Forecast ══════════════════════════════════════════
with tab2:
    st.subheader("📅 30-Day Sales Forecast")

    # Build features for last known row and roll forward
    def make_features_row(series, price=50, on_promo=0):
        s = series.copy()
        row = {}
        for L in (1, 7, 14):
            row[f"lag_{L}"] = s.iloc[-L] if len(s) >= L else s.mean()
        for W in (7, 14):
            row[f"rollmean_{W}"] = s.iloc[-W:].mean() if len(s) >= W else s.mean()
        row["dow"]      = 0   # placeholder
        row["week"]     = 1
        row["price"]    = price
        row["on_promo"] = on_promo
        return pd.DataFrame([row])[feats]

    horizon = 30
    last_sales = sku_df["qty_sold"].values
    avg_price  = sku_df["price"].mean()
    forecasts  = []

    series = list(last_sales)
    for day in range(horizon):
        row_df = make_features_row(pd.Series(series), price=avg_price)
        pred   = max(0, model.predict(row_df)[0])
        forecasts.append(pred)
        series.append(pred)

    future_dates = pd.date_range(sku_df["date"].max() + pd.Timedelta(days=1), periods=horizon)
    forecast_df  = pd.DataFrame({"date": future_dates, "forecast": forecasts})

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(sku_df["date"].iloc[-90:], sku_df["qty_sold"].iloc[-90:],
             label="Actual (last 90d)", color="#4C9BE8")
    ax3.plot(forecast_df["date"], forecast_df["forecast"],
             label="Forecast (30d)", color="#F4845F", linestyle="--", linewidth=2)
    ax3.axvline(sku_df["date"].max(), color="gray", linestyle=":", label="Today")
    ax3.legend(); ax3.set_xlabel("Date"); ax3.set_ylabel("Qty")
    st.pyplot(fig3)

    st.dataframe(forecast_df.set_index("date").round(1), use_container_width=True)

# ══ TAB 3: Inventory Plan ════════════════════════════════════
with tab3:
    st.subheader("📦 Inventory Optimization Recommendation")

    resid_std   = float(sku_df["qty_sold"].std())
    z           = norm.ppf(service_lv / 100)
    mu_L        = float(np.mean(forecasts[:lead_time]) * lead_time)
    sigma_L     = resid_std * (lead_time ** 0.5)
    SS          = z * sigma_L
    ROP         = mu_L + SS
    D_annual    = sku_df["qty_sold"].mean() * 365
    K           = 500    # ordering cost (₹)
    unit_cost   = 100    # ₹ per unit
    hold_rate   = 0.2
    H           = unit_cost * hold_rate
    EOQ         = np.sqrt((2 * D_annual * K) / H) if H > 0 else mu_L
    order_qty   = max(0, ROP - on_hand)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Safety Stock",    f"{SS:.0f} units")
    col2.metric("Reorder Point",   f"{ROP:.0f} units")
    col3.metric("EOQ",             f"{EOQ:.0f} units")
    col4.metric("Order Now?",      f"{order_qty:.0f} units" if on_hand < ROP else "✅ Stock OK")

    if on_hand < ROP:
        st.error(f"⚠️ Stock ({on_hand}) is below Reorder Point ({ROP:.0f}). Place order of {order_qty:.0f} units!")
    else:
        st.success(f"✅ Stock level is healthy. Reorder when stock falls to {ROP:.0f} units.")

    # Summary table
    summary = pd.DataFrame({
        "Metric": ["Lead Time Demand (μ_L)", "Demand Std During Lead Time (σ_L)",
                   "Safety Stock (SS)", "Reorder Point (ROP)", "EOQ", "Recommended Order Qty"],
        "Value":  [f"{mu_L:.1f}", f"{sigma_L:.1f}", f"{SS:.1f}",
                   f"{ROP:.1f}", f"{EOQ:.1f}", f"{order_qty:.1f}"]
    })
    st.table(summary)

    # Export
    csv = forecast_df.to_csv(index=False)
    st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")