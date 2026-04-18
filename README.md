🛒 Retail Sales Forecasting & Inventory Optimization System

A complete, industry-oriented Data Science project that simulates real-world Demand Forecasting + Inventory Replenishment used by modern retail and D2C companies.

This project predicts demand at SKU/store level and converts those predictions into actionable inventory decisions using Safety Stock, Reorder Point, and EOQ.

📌 Problem Statement

Retail businesses lose significant revenue due to:

❌ Stockouts → missed sales
❌ Overstock → high holding costs

This system solves both by:

Forecasting future demand
Modeling uncertainty
Automating replenishment decisions
🎯 Objective
Forecast store/SKU-level demand (daily/weekly)
Quantify forecast uncertainty
Compute Safety Stock
Calculate Reorder Points (ROP)
Suggest Economic Order Quantity (EOQ)

➡️ End-to-end pipeline:
Data → Forecast → Uncertainty → Inventory Policy → Deployment

🏗️ Project Structure
retail_forecast/
├── data/               # Raw & processed datasets
├── notebooks/          # EDA & experiments
├── src/                # Core pipeline scripts
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── inventory_optimization.py
├── models/             # Saved ML models (.pkl)
├── outputs/            # Predictions & recommendations
├── images/             # Charts & visualizations
├── reports/            # Optional reports
├── docs/               # Documentation
├── app/
│   ├── app_streamlit.py   # Interactive dashboard
│   └── api.py             # FastAPI service
├── main.py             # End-to-end pipeline runner
├── requirements.txt
└── .gitignore
⚙️ Tech Stack
Component	Tools
Language	Python 3.10+
Data	Pandas, NumPy
ML Models	Random Forest, XGBoost
Statistics	SciPy, Statsmodels
Visualization	Matplotlib, Seaborn
Dashboard	Streamlit
API	FastAPI + Uvicorn
Model Storage	Joblib
🚀 Quick Start
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/retail-forecast.git
cd retail-forecast
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Full Pipeline
python main.py
4️⃣ Launch Dashboard
streamlit run app/app_streamlit.py

👉 Open in browser:
http://localhost:8501

5️⃣ Run API
uvicorn app.api:app --reload
Test API
curl -X POST http://127.0.0.1:8000/replenishment \
-H "Content-Type: application/json" \
-d '{"store_id":"S01","item_id":"ITEM_A","on_hand":100,"lead_time":7}'
📊 Pipeline Workflow
Step	Description	Output
1	Data Generation	data/retail_timeseries.csv
2	Preprocessing	data/retail_clean.csv
3	EDA	images/*.png
4	Feature Engineering	data/retail_features.csv
5	Model Training	models/*.pkl
6	Inventory Optimization	outputs/inventory_recommendations.csv
