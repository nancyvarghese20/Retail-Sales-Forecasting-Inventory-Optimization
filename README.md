# 🛒 Retail Sales Forecasting & Inventory Optimization System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=flat-square&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A complete, end-to-end Data Science project for retail demand forecasting and inventory replenishment — built for placements, internships, and GitHub proof-of-work.

---

## 📌 Problem Statement

Retail businesses lose crores annually from two inventory extremes:

- **Stockouts** — lost sales when shelves run empty
- **Overstock** — high holding costs from excess inventory

This system forecasts future demand per SKU-Store combination and computes optimal **Safety Stock**, **Reorder Points**, and **Economic Order Quantities** — automating the replenishment decision with ML-driven precision.



## ⚙️ Tech Stack

| Component | Tool / Library |
|-----------|---------------|
| Language | Python 3.10+ |
| Data wrangling | Pandas, NumPy |
| ML Models | Scikit-learn (Random Forest), XGBoost |
| Statistics | SciPy, Statsmodels |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| REST API | FastAPI + Uvicorn |
| Model storage | Joblib (.pkl files) |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/retail-sales-forecasting-inventory-optimization.git
cd retail-sales-forecasting-inventory-optimization
pip install -r requirements.txt
```

### 2. Run Full Pipeline Locally

```bash
python main.py
```

This runs all 6 steps in order and generates:

- `data/retail_timeseries.csv`, `retail_clean.csv`, `retail_features.csv`
- `models/rf_model.pkl`, `models/resid_info.pkl`
- `outputs/model_comparison.csv`, `outputs/inventory_recommendations.csv`
- `images/01–09_*.png` (all EDA and model charts)

### 3. Launch Streamlit Dashboard

```bash
streamlit run app/app_streamlit.py
```

### 4. Launch REST API

```bash
uvicorn app.api:app --reload
```

Test with curl:

```bash
curl -X POST http://127.0.0.1:8000/replenishment \
  -H "Content-Type: application/json" \
  -d '{"store_id":"S01","item_id":"ITEM_A","on_hand":100,"lead_time":7,"service_level":0.95}'
```

---

## 📊 Pipeline — Step by Step

| # | Script | Purpose | Output |
|---|--------|---------|--------|
| 1 | `src/generate_data.py` | Synthetic data (3 stores × 5 SKUs, 2 years) | `data/retail_timeseries.csv` |
| 2 | `src/preprocess.py` | Clean, gap-fill, add calendar features | `data/retail_clean.csv` |
| 3 | `src/eda.py` | 6 EDA charts (trends, heatmaps, promo analysis) | `images/01–06_*.png` |
| 4 | `src/feature_engineering.py` | Lag & rolling window features | `data/retail_features.csv` |
| 5 | `src/train_model.py` | Train RF + XGBoost, evaluate, save | `models/*.pkl`, `outputs/model_comparison.csv` |
| 6 | `src/inventory_optimization.py` | Compute SS, ROP, EOQ per SKU-Store | `outputs/inventory_recommendations.csv` |



## 📈 Model Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | ~3.2 | ~4.8 | ~0.91 |
| XGBoost | ~3.0 | ~4.5 | ~0.92 |
| Seasonal Naive (lag-7) | ~7.5 | ~10.1 | ~0.65 |

> Exact numbers depend on random seed and data split.

---

## ☁️ Streamlit Cloud Deployment

### Step 1 — Generate & push model files

Run the pipeline locally, then push the artifacts to GitHub:

```bash
python main.py

# Remove *.pkl lines from .gitignore, then:
git add data/ models/ app/app_streamlit.py
git commit -m "Add pre-generated model artifacts for Streamlit Cloud"
git push origin main
```

### Step 2 — Configure on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
2. Repository: your GitHub repo
3. Branch: `main`
4. **Main file path: `app/app_streamlit.py`** ← Important

### Step 3 — Path fix in `app_streamlit.py`

These two lines must be at the top of the file to fix `ModuleNotFoundError` on Cloud:

```python
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
```

---

## 🔌 REST API Reference

**`POST /replenishment`**

Request body:

```json
{
  "store_id": "S01",
  "item_id": "ITEM_A",
  "on_hand": 100,
  "lead_time": 7,
  "service_level": 0.95
}
```

Response:

```json
{
  "store_id": "S01",
  "item_id": "ITEM_A",
  "mu_L": 312.5,
  "safety_stock": 42.3,
  "reorder_point": 354.8,
  "EOQ": 500.0,
  "order_qty": 500.0,
  "order_value_INR": 25000.0,
  "alert": "⚠️ Stock below ROP — order immediately!"
}
```

**`GET /`** — health check



## 📁 Key Output Files

| File | Description |
|------|-------------|
| `outputs/model_comparison.csv` | MAE, RMSE, R² for all models |
| `outputs/inventory_recommendations.csv` | SS, ROP, EOQ, Order Qty for all 15 SKU-stores |
| `images/07_feature_importance.png` | Top-20 feature importances (RF) |
| `images/08_actual_vs_predicted.png` | Actual vs Predicted scatter plot |
| `images/09_order_value_by_sku.png` | Recommended order value per SKU |

---

