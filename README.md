🛒 Retail Sales Forecasting & Inventory Optimization System

A complete, industry-oriented Data Science project designed for placements, internships, and strong GitHub proof-of-work.

This project simulates a real-world Demand Forecasting + Inventory Replenishment pipeline, used by modern retail and D2C companies to minimize stockouts and reduce excess inventory.

📌 Objective
Function	Output
Forecast store/SKU-level demand	Daily / weekly sales predictions
Capture demand uncertainty	Residual standard deviation
Compute safety stock	Based on service-level targets
Calculate reorder points	Prevent stockouts during lead time
Optimize replenishment (EOQ)	Cost-efficient order quantity

This system delivers a complete pipeline from demand prediction → inventory decision automation, combining Data Science with Operations Research.

🌍 Industry Relevance

Retailers lose significant revenue due to:

❌ Stockouts → missed sales
❌ Overstock → high holding costs

This project models how real supply chain systems work:
Data → Forecasting → Uncertainty → Inventory Policy → Deployment

🧑‍💼 Practical Applications
Replenishment automation
Fill-rate optimization
Working capital efficiency
Multi-SKU inventory planning
D2C / FMCG / Grocery retail systems
🏗️ Project Structure
retail_forecast/
├── data/               ← Raw & processed datasets
├── notebooks/          ← EDA & experimentation
├── src/                ← Core pipeline scripts
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── inventory_optimization.py
├── models/             ← Saved ML models (.pkl)
├── outputs/            ← Predictions & recommendations
├── images/             ← Charts & visualizations
├── reports/            ← Optional reports
├── docs/               ← Documentation
├── app/
│   ├── app_streamlit.py   ← Planner dashboard
│   └── api.py             ← FastAPI service
├── main.py             ← End-to-end pipeline runner
├── requirements.txt
└── .gitignore
⚙️ Tech Stack
Component	Tools
Language	Python 3.10+
Data Processing	Pandas, NumPy
ML Models	Random Forest, XGBoost
Statistics	SciPy, Statsmodels
Visualization	Matplotlib, Seaborn
Dashboard	Streamlit
API	FastAPI + Uvicorn
Model Persistence	Joblib
🚀 Run the Project Locally
1️⃣ Clone & Install
git clone https://github.com/YOUR_USERNAME/retail-forecast.git
cd retail-forecast
pip install -r requirements.txt
2️⃣ Run Full Pipeline
python main.py
3️⃣ Launch Dashboard
streamlit run app/app_streamlit.py

Access UI at:
👉 http://localhost:8501

4️⃣ Run REST API
uvicorn app.api:app --reload
Test API
curl -X POST http://127.0.0.1:8000/replenishment \
-H "Content-Type: application/json" \
-d '{"store_id":"S01","item_id":"ITEM_A","on_hand":100,"lead_time":7}'
📊 Pipeline Workflow
Step	Script	Output
Data generation	src/generate_data.py	data/retail_timeseries.csv
Preprocessing	src/preprocess.py	data/retail_clean.csv
EDA	src/eda.py	images/*.png
Feature engineering	src/feature_engineering.py	data/retail_features.csv
Model training	src/train_model.py	models/*.pkl
Inventory optimization	src/inventory_optimization.py	outputs/inventory_recommendations.csv
