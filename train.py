"""
train.py — Batch model trainer for Retention Analytics
------------------------------------------------------
- Trains Logistic Regression, Random Forest, XGBoost, and MLP
- Uses sample.csv as input dataset
- Saves trained models to /models
- Saves predictions, metrics, and segmentation outputs to /outputs
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils.load import load_dataset, minimal_clean, detect_binary_target_candidates
from utils.retain import (
    split_features_target,
    build_model_pipeline,
    fit_model,
    predict_proba,
    compute_metrics,
    estimate_clv,
    clv_portfolio_metrics,
    segment_customers,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_PATH = os.path.join("data", "sample.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODELS = {
    "lr": {},
    "rf": {"n_estimators": 300, "max_depth": 12},
    "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    "mlp": {"hidden_units": 64, "hidden_layers": 2, "learning_rate_init": 0.001},
}
THRESHOLD = 0.5
GROSS_MARGIN = 0.6
CLV_CAP = 60
DEFAULT_ARPU = 50.0

# ---------------------------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------------------------
print(f"Loading dataset from {DATA_PATH}")
raw = load_dataset(path=DATA_PATH)
df = minimal_clean(raw)
target_col = detect_binary_target_candidates(df)[0]
positive_label = None

arpu_col = "MonthlyCharges" if "MonthlyCharges" in df.columns else None

# ---------------------------------------------------------------------
# Training all models
# ---------------------------------------------------------------------
for key, params in MODELS.items():
    print(f"\n🔹 Training model: {key.upper()}")
    X, y = split_features_target(df, target_col, positive_label)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe, _, _ = build_model_pipeline(X_tr, key, params)
    pipe = fit_model(pipe, X_tr, y_tr)
    print("✅ Model fitted.")

    val_prob = predict_proba(pipe, X_val)
    auc = roc_auc_score(y_val, val_prob)
    if auc < 0.5:
        print("⚠️  Detected inverted probabilities — correcting polarity.")
        val_prob = 1 - val_prob

    metrics = compute_metrics(y_val.values, val_prob, thr=THRESHOLD)
    print("Validation metrics:", metrics)

    all_prob = predict_proba(pipe, X)
    if auc < 0.5:
        all_prob = 1 - all_prob

    result_df = df.copy()
    result_df["churn_prob"] = all_prob
    result_df["predicted_clv"] = estimate_clv(
        result_df,
        churn_prob_col="churn_prob",
        arpu_col=arpu_col,
        default_arpu=DEFAULT_ARPU,
        margin=GROSS_MARGIN,
        max_months=CLV_CAP,
    )

    risk_thr = float(result_df["churn_prob"].median())
    value_thr = float(result_df["predicted_clv"].median())
    result_df["segment"] = segment_customers(result_df, "churn_prob", "predicted_clv", risk_thr, value_thr)
    portfolio = clv_portfolio_metrics(result_df, "predicted_clv", "churn_prob")

    # Save model + results
    model_file = os.path.join(MODELS_DIR, f"{key}_model.pkl")
    joblib.dump(pipe, model_file)
    result_df.to_csv(os.path.join(OUTPUTS_DIR, f"{key}_predictions.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUTS_DIR, f"{key}_metrics.csv"), index=False)
    pd.DataFrame([portfolio]).to_csv(os.path.join(OUTPUTS_DIR, f"{key}_portfolio.csv"), index=False)

    print(f"💾 Saved model → {model_file}")
    print(f"📊 Outputs saved to → {OUTPUTS_DIR}/")

print("\n✅ All models trained successfully.")
