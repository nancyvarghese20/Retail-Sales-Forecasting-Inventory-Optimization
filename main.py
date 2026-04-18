

import subprocess
import sys

STEPS = [
    ("Step 1 — Generate synthetic data",     "src/generate_data.py"),
    ("Step 2 — Preprocess & clean",          "src/preprocess.py"),
    ("Step 3 — Exploratory Data Analysis",   "src/eda.py"),
    ("Step 4 — Feature engineering",         "src/feature_engineering.py"),
    ("Step 5 — Train & evaluate models",     "src/train_model.py"),
    ("Step 6 — Inventory optimization",      "src/inventory_optimization.py"),
]

def run_step(label: str, script: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"\n❌  {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅  Done: {label}")

if __name__ == "__main__":
    print("\n🚀  Starting Retail Sales Forecasting & Inventory Optimization Pipeline\n")
    for label, script in STEPS:
        run_step(label, script)

    print("\n" + "="*60)
    print("  ✅  ALL STEPS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  • View charts      →  images/")
    print("  • Model metrics    →  outputs/model_comparison.csv")
    print("  • PO reco          →  outputs/inventory_recommendations.csv")
    print("  • Launch dashboard →  streamlit run app/app_streamlit.py")
    print("  • Launch API       →  uvicorn app.api:app --reload")
