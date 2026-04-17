# 🛒 Retail Sales Forecasting & Inventory Optimization System

> A complete, industry-oriented Data Science project for placements, internships, and GitHub proof-of-work.

---

## 📌 Problem Statement

Retail businesses lose crores annually from **stockouts** (missed sales) and **overstock** (high holding costs).  
This system forecasts future demand and computes optimal **Safety Stock**, **Reorder Points**, and **Economic Order Quantities** — automating the replenishment decision.

---

## 🏗️ Project Structure

```
retail_forecast/
├── data/               ← Raw & processed CSV datasets
├── notebooks/          ← Jupyter notebooks for exploration
├── src/                ← Python pipeline scripts
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── inventory_optimization.py
├── models/             ← Saved ML model artifacts (.pkl)
├── outputs/            ← CSV results (recommendations, metrics)
├── images/             ← Charts & visualizations
├── reports/            ← PDF/HTML reports (optional)
├── docs/               ← Documentation
├── app/
│   ├── app_streamlit.py   ← Interactive planner dashboard
│   └── api.py             ← FastAPI replenishment endpoint
├── main.py             ← Full pipeline orchestrator
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| ML Models | Scikit-learn (Random Forest), XGBoost |
| Statistics | SciPy, Statsmodels |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| REST API | FastAPI + Uvicorn |
| Model persistence | Joblib |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/retail-forecast.git
cd retail-forecast
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python main.py
```

### 3. Launch Streamlit dashboard

```bash
streamlit run app/app_streamlit.py
```

### 4. Launch REST API

```bash
uvicorn app.api:app --reload
# Test: curl -X POST http://127.0.0.1:8000/replenishment \
#   -H "Content-Type: application/json" \
#   -d '{"store_id":"S01","item_id":"ITEM_A","on_hand":100,"lead_time":7}'
```

---

## 📊 Pipeline Steps

| Step | Script | Output |
|------|--------|--------|
| 1. Generate data | `src/generate_data.py` | `data/retail_timeseries.csv` |
| 2. Preprocess | `src/preprocess.py` | `data/retail_clean.csv` |
| 3. EDA | `src/eda.py` | `images/01–06_*.png` |
| 4. Feature engineering | `src/feature_engineering.py` | `data/retail_features.csv` |
| 5. Train models | `src/train_model.py` | `models/rf_model.pkl`, `outputs/model_comparison.csv` |
| 6. Inventory optimization | `src/inventory_optimization.py` | `outputs/inventory_recommendations.csv` |

---

## 📐 Inventory Math

| Formula | Meaning |
|---------|---------|
| `SS = z × σ_L` | Safety Stock (z = service-level z-score) |
| `ROP = μ_L + SS` | Reorder Point (μ_L = mean demand during lead time) |
| `EOQ = √(2DK/H)` | Economic Order Qty (D=annual demand, K=order cost, H=holding cost) |
| `Q = max(EOQ, ROP − on_hand)` | Final recommended order quantity |

---

## 🎯 Interview Talking Points

1. **Objective**: Predict demand → maintain optimal stock → reduce stockouts & overstock.
2. **Data**: Simulated 2-year daily sales across 3 stores × 5 SKUs with seasonality, promos, stockouts.
3. **Model**: Random Forest Regressor with lag/rolling features; compared to XGBoost & seasonal naive.
4. **Inventory logic**: Service-level-driven SS/ROP + EOQ for cost-optimal ordering.
5. **Deployment**: Streamlit dashboard for planners + FastAPI for system integration.

---

## 📈 Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | ~3.2 | ~4.8 | ~0.91 |
| XGBoost | ~3.0 | ~4.5 | ~0.92 |
| Seasonal Naive | ~7.5 | ~10.1 | ~0.65 |

*(Exact numbers depend on random seed and data split)*

---

## 🧑‍💼 Use Cases

- Grocery & FMCG replenishment
- E-commerce (Flipkart, Amazon seller tools)
- Fashion (seasonal demand)
- Pharma (expiry-aware stock management)

---

*Built as a portfolio project for Data Analyst / Data Science roles.*
