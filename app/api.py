"""
STEP 8 — FastAPI Replenishment API
File: app/api.py

Exposes a POST /replenishment endpoint so downstream systems
(ERP, WMS, other microservices) can request PO recommendations.

Run: uvicorn app.api:app --reload
Test with curl:
  curl -X POST http://127.0.0.1:8000/replenishment \
       -H "Content-Type: application/json" \
       -d '{"store_id":"S01","item_id":"ITEM_A","on_hand":100,"lead_time":7,"service_level":0.95}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

app = FastAPI(title="Retail Replenishment API", version="1.0")

# ── Load artefacts at startup ─────────────────────────────────────────────────
artifact   = joblib.load("models/rf_model.pkl")
rf         = artifact["model"]
feat_cols  = artifact["features"]
resid_info = joblib.load("models/resid_info.pkl")
resid_std  = resid_info["resid_std"]
df_all     = pd.read_csv("data/retail_features.csv", parse_dates=["date"])

# ── Request / Response schemas ────────────────────────────────────────────────
class ReplenishRequest(BaseModel):
    store_id:      str   = Field(..., example="S01")
    item_id:       str   = Field(..., example="ITEM_A")
    on_hand:       float = Field(..., ge=0, example=100)
    lead_time:     int   = Field(7, ge=1, le=90)
    service_level: float = Field(0.95, ge=0.80, le=0.999)

class ReplenishResponse(BaseModel):
    store_id:      str
    item_id:       str
    mu_L:          float
    safety_stock:  float
    reorder_point: float
    EOQ:           float
    order_qty:     float
    order_value_INR: float
    alert:         str

# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/replenishment", response_model=ReplenishResponse)
def replenishment(req: ReplenishRequest):
    grp = df_all[
        (df_all["store_id"] == req.store_id) &
        (df_all["item_id"]  == req.item_id)
    ].sort_values("date").tail(30)

    if grp.empty:
        raise HTTPException(status_code=404, detail="SKU-Store combination not found.")

    X    = grp[feat_cols]
    fcst = np.maximum(0, rf.predict(X))

    z       = norm.ppf(req.service_level)
    mu_L    = float(fcst[:req.lead_time].sum())
    sigma_L = float(resid_std * (req.lead_time ** 0.5))
    SS      = float(z * sigma_L)
    ROP     = float(mu_L + SS)

    unit_cost    = float(grp["unit_cost"].iloc[0])
    ordering_cost = float(grp["ordering_cost"].iloc[0])
    holding_rate = float(grp["holding_rate"].iloc[0])
    D_annual     = float(np.mean(fcst) * 365)
    H            = unit_cost * holding_rate
    EOQ          = float(np.sqrt((2 * D_annual * ordering_cost) / H)) if H > 0 else mu_L
    Q            = float(max(0.0, max(EOQ, ROP - req.on_hand)))

    alert = "⚠️ Stock below ROP — order immediately!" if req.on_hand <= ROP else "✅ Stock OK"

    return ReplenishResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        mu_L=round(mu_L, 1),
        safety_stock=round(SS, 1),
        reorder_point=round(ROP, 1),
        EOQ=round(EOQ, 1),
        order_qty=round(Q, 1),
        order_value_INR=round(Q * unit_cost, 2),
        alert=alert,
    )

@app.get("/")
def root():
    return {"message": "Retail Replenishment API is running. POST to /replenishment"}
