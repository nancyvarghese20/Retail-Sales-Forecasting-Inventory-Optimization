
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
from scipy.stats import norm

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Forecast & Replenishment",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Sales Forecasting & Inventory Optimization")
st.markdown("---")

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    artifact   = joblib.load("models/rf_model.pkl")
    resid_info = joblib.load("models/resid_info.pkl")
    return artifact["model"], artifact["features"], resid_info["resid_std"]

@st.cache_data
def load_data():
    return pd.read_csv("data/retail_features.csv", parse_dates=["date"])

rf, feat_cols, resid_std = load_model()
df = load_data()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("📦 SKU Selection")
stores = sorted(df["store_id"].unique())
items  = sorted(df["item_id"].unique())

store    = st.sidebar.selectbox("Store", stores)
item     = st.sidebar.selectbox("Product (SKU)", items)
on_hand  = st.sidebar.number_input("On-Hand Inventory (units)", min_value=0, value=200)
lead_time = st.sidebar.slider("Lead Time (days)", min_value=1, max_value=30, value=7)
service_level = st.sidebar.slider("Service Level (%)", min_value=80, max_value=99, value=95)

# ── Filter data for selected SKU-store ───────────────────────────────────────
grp = df[(df["store_id"] == store) & (df["item_id"] == item)].copy()
grp = grp.sort_values("date").tail(90)   # use last 90 days

# ── Forecast ──────────────────────────────────────────────────────────────────
X_grp = grp[feat_cols]
grp["forecast"] = np.maximum(0, rf.predict(X_grp))

# ── Layout: 2 columns ─────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 Forecast vs Actual — {store} | {item}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grp["date"], grp["qty_sold"],  label="Actual",   linewidth=1.2)
    ax.plot(grp["date"], grp["forecast"],  label="Forecast", linewidth=1.2, linestyle="--", color="orange")
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Inventory Recommendation")

    z       = norm.ppf(service_level / 100)
    mu_L    = grp["forecast"].head(lead_time).sum()
    sigma_L = resid_std * (lead_time ** 0.5)
    SS      = z * sigma_L
    ROP     = mu_L + SS

    unit_cost    = float(grp["unit_cost"].iloc[0])
    ordering_cost = float(grp["ordering_cost"].iloc[0])
    holding_rate = float(grp["holding_rate"].iloc[0])
    D_annual     = grp["forecast"].mean() * 365
    H            = unit_cost * holding_rate
    EOQ          = np.sqrt((2 * D_annual * ordering_cost) / H) if H > 0 else mu_L
    Q            = max(0.0, max(EOQ, ROP - on_hand))

    st.metric("Safety Stock (SS)", f"{SS:.0f} units")
    st.metric("Reorder Point (ROP)", f"{ROP:.0f} units")
    st.metric("Economic Order Qty (EOQ)", f"{EOQ:.0f} units")
    st.metric("🚚 Recommended Order Qty", f"{Q:.0f} units",
              delta=f"₹ {Q*unit_cost:,.0f} value")

    if on_hand <= ROP:
        st.error("⚠️ Stock below Reorder Point — place order NOW!")
    else:
        st.success("✅ Stock level is adequate.")

# ── Export PO ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 Export Purchase Order")

po_data = {
    "store_id": store, "item_id": item,
    "on_hand": on_hand, "lead_time_days": lead_time,
    "service_level_pct": service_level,
    "safety_stock": round(SS, 1), "reorder_point": round(ROP, 1),
    "EOQ": round(EOQ, 1), "order_qty": round(Q, 1),
    "unit_cost_INR": unit_cost, "order_value_INR": round(Q * unit_cost, 2),
}
po_df = pd.DataFrame([po_data])
csv_buf = io.StringIO()
po_df.to_csv(csv_buf, index=False)

st.dataframe(po_df)
st.download_button(
    label="⬇️ Download PO as CSV",
    data=csv_buf.getvalue(),
    file_name=f"PO_{store}_{item}.csv",
    mime="text/csv",
)

# ── Historical stats ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗃️ Historical Sales Summary (last 90 days)")
stats = grp["qty_sold"].describe().rename("Value").to_frame()
st.dataframe(stats.T)
