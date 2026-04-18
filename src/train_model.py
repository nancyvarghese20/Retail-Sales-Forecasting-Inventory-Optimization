

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — skipping XGB model.")

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/retail_features.csv", parse_dates=["date"])

# ── Define feature columns ────────────────────────────────────────────────────
EXCLUDE = ["qty_sold", "date", "store_id", "item_id",
           "stockout_flag", "year_month"]
feat_cols = [c for c in df.columns if c not in EXCLUDE]
TARGET    = "qty_sold"

X = df[feat_cols]
y = df[TARGET]
groups = df["store_id"] + "_" + df["item_id"]   # group key for split

# ── Group-aware train/test split (80/20) ──────────────────────────────────────
gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

# ── Helper: evaluate model ────────────────────────────────────────────────────
def evaluate(model_name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MASE-like: compare to seasonal naive (lag-7 on test set)
    print(f"\n── {model_name} ──────────────────────")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²  : {r2:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []

# ── Model 1: Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest …")
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
rf_pred = np.maximum(0, rf.predict(X_test))
results.append(evaluate("Random Forest", y_test, rf_pred))

joblib.dump({"model": rf, "features": feat_cols}, "models/rf_model.pkl")
print("Saved → models/rf_model.pkl")

# ── Model 2: XGBoost ─────────────────────────────────────────────────────────
if XGB_AVAILABLE:
    print("\nTraining XGBoost …")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    xgb_pred = np.maximum(0, xgb_model.predict(X_test))
    results.append(evaluate("XGBoost", y_test, xgb_pred))
    joblib.dump({"model": xgb_model, "features": feat_cols}, "models/xgb_model.pkl")
    print("Saved → models/xgb_model.pkl")

# ── Baseline: Seasonal Naive (lag-7) ─────────────────────────────────────────
lag7_col = "lag_7"
if lag7_col in X_test.columns:
    naive_pred = np.maximum(0, X_test[lag7_col].values)
    results.append(evaluate("Seasonal Naive (lag-7)", y_test, naive_pred))

# ── Save comparison table ─────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

# ── Feature Importance plot (RF) ──────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
top20 = fi.head(20)
fig, ax = plt.subplots(figsize=(10, 6))
top20.plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Top-20 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("images/07_feature_importance.png", dpi=120)
plt.close()
print("Saved → images/07_feature_importance.png")

# ── Actual vs Predicted plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(y_test[:500], rf_pred[:500], alpha=0.3, s=10, color="steelblue")
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted (Random Forest) — first 500 test samples")
lims = [0, max(y_test.max(), rf_pred.max()) * 1.05]
ax.plot(lims, lims, "r--", linewidth=1)
plt.tight_layout()
plt.savefig("images/08_actual_vs_predicted.png", dpi=120)
plt.close()
print("Saved → images/08_actual_vs_predicted.png")

# ── Save residuals for inventory step ────────────────────────────────────────
resid_std = float(np.std(y_test.values - rf_pred))
joblib.dump({"resid_std": resid_std, "feat_cols": feat_cols}, "models/resid_info.pkl")
print(f"\nResidual std: {resid_std:.2f}  →  saved to models/resid_info.pkl")
