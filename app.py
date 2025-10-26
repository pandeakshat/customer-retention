"""
Retention • Churn & Value Analytics (Streamlit)
------------------------------------------------
Lightweight, Chrome-independent implementation.

Features:
- 4 model options (LR, RF, XGB, MLP)
- Model training / inference pipeline
- Pretrained model toggle for sample.csv
- Churn, CLV, segmentation & recommendations
- Multi-tab analysis + printable report view
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

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
    get_feature_importance_df,
)
from utils.report import (
    plot_churn_density,
    plot_feature_importance,
    plot_risk_value_quadrant,
    plot_clv_distribution,
    plot_retention_curve,
    build_summary_table,
    generate_recommendations,
)

# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Retention • Churn & Value Analytics", layout="wide")
st.title("Retention • Churn & Value Analytics")

# ---------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------
defaults = {
    "trained": False,
    "result_df": pd.DataFrame(),
    "imp_df": None,
    "metrics": {},
    "portfolio": {},
    "risk_thr": 0.5,
    "value_thr": 0.0,
    "y_val": pd.Series(dtype=float),
    "val_prob": np.array([]),
    "last_model": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------
# Sidebar — data + parameters
# ---------------------------------------------------------------------
st.sidebar.header("Dataset & Configuration")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
default_path = os.path.join("data", "sample.csv")
st.sidebar.caption(f"Default path: `{default_path}`")

# ---------------------------------------------------------------------
# Run Mode Selector
# ---------------------------------------------------------------------
st.sidebar.header("Run Mode")
run_mode = st.sidebar.radio(
    "Choose Mode",
    ["Use Pre-trained Model", "Train / Retrain Model"],
    index=1 if uploaded is not None else 0,
)

use_pretrained = (run_mode == "Use Pre-trained Model") and (uploaded is None)
run_clicked = (run_mode == "Train / Retrain Model")

if run_mode == "Use Pre-trained Model":
    st.sidebar.info("Uses models from `/models/`. Uploading a dataset disables this mode.")
else:
    st.sidebar.info("This will train a fresh model. May take a few minutes for RF/XGB.")

@st.cache_data(show_spinner=False)
def _load_clean(_uploaded):
    raw = load_dataset(path=default_path, uploaded_file=_uploaded)
    clean = minimal_clean(raw)
    return raw, clean

try:
    raw_df, df = _load_clean(uploaded)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

candidate_targets = detect_binary_target_candidates(df)
target_col = st.sidebar.selectbox(
    "Target column (binary)",
    options=list(df.columns),
    index=(list(df.columns).index(candidate_targets[0]) if candidate_targets else 0),
)
positive_label = None
if df[target_col].nunique(dropna=True) > 2 or df[target_col].dtype == "O":
    unique_vals = df[target_col].dropna().astype(str).unique().tolist()
    positive_label = st.sidebar.selectbox("Positive label (maps to 1)", options=unique_vals)

arpu_col_opt = ["<none>"] + list(df.columns)
arpu_col = st.sidebar.selectbox(
    "Revenue / ARPU column",
    options=arpu_col_opt,
    index=(arpu_col_opt.index("MonthlyCharges") if "MonthlyCharges" in df.columns else 0),
)
arpu_col = None if arpu_col == "<none>" else arpu_col

tenure_col_opt = ["<none>"] + list(df.columns)
tenure_col = st.sidebar.selectbox(
    "Tenure column (for retention curve)",
    options=tenure_col_opt,
    index=(tenure_col_opt.index("tenure") if "tenure" in df.columns else 0),
)
tenure_col = None if tenure_col == "<none>" else tenure_col

# ---------------------------------------------------------------------
# Model selection + params
# ---------------------------------------------------------------------
st.sidebar.header("Model Selection")
model_label = st.sidebar.radio(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network (MLP)"],
    index=0,
)
model_map = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb", "Neural Network (MLP)": "mlp"}
model_key = model_map[model_label]

params = {}
if model_key == "lr":
    params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
elif model_key == "rf":
    params["n_estimators"] = st.sidebar.slider("Trees", 50, 1000, 300, 50)
    params["max_depth"] = st.sidebar.slider("Max Depth", 2, 40, 12, 1)
elif model_key == "xgb":
    params["n_estimators"] = st.sidebar.slider("Trees", 50, 1000, 300, 50)
    params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    params["max_depth"] = st.sidebar.slider("Max Depth", 2, 12, 6, 1)
elif model_key == "mlp":
    params["hidden_units"] = st.sidebar.slider("Hidden Units", 16, 256, 64, 16)
    params["hidden_layers"] = st.sidebar.slider("Hidden Layers", 1, 4, 2, 1)
    params["learning_rate_init"] = st.sidebar.slider("Learning Rate", 0.0001, 0.05, 0.001, 0.0001)

st.sidebar.header("Threshold & CLV")
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, 0.01)
gross_margin = st.sidebar.slider("Gross Margin", 0.1, 0.95, 0.6, 0.05)
clv_cap = st.sidebar.slider("Max Expected Months", 6, 120, 60, 1)
default_arpu = st.sidebar.number_input("Default ARPU", value=50.0, min_value=0.0, step=1.0)

# ---------------------------------------------------------------------
# Run mode logic
# ---------------------------------------------------------------------
import joblib

if use_pretrained:
    # ---------------- Pre-trained Model Path ----------------
    with st.spinner(f"Loading pre-trained {model_label}..."):
        try:
            model_file = os.path.join("models", f"{model_key}_model.pkl")
            if not os.path.exists(model_file):
                st.error(f"Pre-trained {model_label} not found in /models/. Please train first.")
                st.stop()

            pipe = joblib.load(model_file)
            st.success(f"Loaded pre-trained {model_label} from /models/")

            X, y = split_features_target(df, target_col, positive_label)
            y_val = y
            val_prob = predict_proba(pipe, X)

            auc = roc_auc_score(y_val, val_prob)
            if auc < 0.5:
                st.warning("Detected inverted probabilities — correcting label polarity.")
                val_prob = 1 - val_prob

        except Exception as e:
            st.error(f"Failed to load pre-trained model: {e}")
            st.stop()

else:
    # ---------------- Train / Retrain Path ----------------
    if st.button("🚀 Start Training"):
        progress = st.progress(0, text="Initializing training...")
        try:
            X, y = split_features_target(df, target_col, positive_label)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            progress.progress(10, text="Building model pipeline...")
            pipe, _, _ = build_model_pipeline(X_tr, model_key, params)

            progress.progress(30, text="Fitting model...")
            pipe = fit_model(pipe, X_tr, y_tr)

            progress.progress(60, text="Scoring validation data...")
            val_prob = predict_proba(pipe, X_val)
            auc = roc_auc_score(y_val, val_prob)
            if auc < 0.5:
                st.warning("Detected inverted probabilities — correcting label polarity.")
                val_prob = 1 - val_prob

            progress.progress(80, text="Saving trained model...")
            model_file = os.path.join("models", f"{model_key}_model.pkl")
            joblib.dump(pipe, model_file)

            st.success(f"Model saved to {model_file}")
            progress.progress(100, text="Training complete!")

        except Exception as e:
            st.error(f"Error during training: {e}")
            st.stop()
    else:
        st.info("Click **Start Training** to build your model.")
        st.stop()

# ---------------------------------------------------------------------
# Common post-run logic (metrics, CLV, visualizations)
# ---------------------------------------------------------------------
metrics = compute_metrics(y_val.values, val_prob, thr=threshold)

try:
    inner_model = pipe.named_steps.get("clf")
    feature_names = (
        pipe.named_steps["pre"].get_feature_names_out()
        if hasattr(pipe.named_steps["pre"], "get_feature_names_out")
        else list(df.columns)
    )
    imp_df = get_feature_importance_df(inner_model, feature_names)
except Exception:
    imp_df = None

all_prob = predict_proba(pipe, df.drop(columns=[target_col], errors="ignore"))
result_df = df.copy()
result_df["churn_prob"] = all_prob
result_df["predicted_clv"] = estimate_clv(
    result_df,
    churn_prob_col="churn_prob",
    arpu_col=arpu_col,
    default_arpu=default_arpu,
    margin=gross_margin,
    max_months=clv_cap,
)
risk_thr = float(result_df["churn_prob"].median())
value_thr = float(result_df["predicted_clv"].median())
result_df["segment"] = segment_customers(result_df, "churn_prob", "predicted_clv", risk_thr, value_thr)
portfolio = clv_portfolio_metrics(result_df, "predicted_clv", "churn_prob")

# ---------------------------------------------------------------------
# Visualization Tabs
# ---------------------------------------------------------------------
summary_df = build_summary_table(metrics, portfolio)
fig_density = plot_churn_density(result_df, "churn_prob", threshold)
fig_importance = plot_feature_importance(imp_df, 20) if imp_df is not None else None
fig_quadrant = plot_risk_value_quadrant(result_df, "churn_prob", "predicted_clv", "segment", risk_thr, value_thr)
fig_clv = plot_clv_distribution(result_df, "predicted_clv")
fig_retention = plot_retention_curve(result_df.join(y_val.rename("target")), tenure_col, "target")

cm = confusion_matrix(y_val, (val_prob >= threshold).astype(int))
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
top_risk_df = result_df.sort_values("churn_prob", ascending=False).head(50)
top_value_df = result_df.sort_values("predicted_clv", ascending=False).head(50)
top_value_df["Value Share %"] = top_value_df["predicted_clv"] / result_df["predicted_clv"].sum() * 100
seg_df = result_df["segment"].value_counts().rename_axis("Segment").reset_index(name="Count")

# ---------------------------------------------------------------------
# Tabs (with unique chart keys)
# ---------------------------------------------------------------------
tabs = st.tabs([
    "Summary", "Churn Density", "Feature Importance",
    "Risk–Value Quadrant", "CLV Distribution", "Retention Curve",
    "Top At-Risk Customers", "Top Customers by Value", "Recommendations",
    "All Plots & Tables", "Future Steps"
])

with tabs[0]:
    st.subheader("Model Summary")
    st.table(summary_df)

with tabs[1]:
    st.subheader("Churn Probability — Density")
    st.plotly_chart(fig_density, config={"responsive": True, "displaylogo": False}, key="density_tab")
    st.dataframe(cm_df, width='stretch')

with tabs[2]:
    st.subheader("Feature Importance")
    if imp_df is not None and not imp_df.empty:
        fig_importance = plot_feature_importance(imp_df, 20)
        st.plotly_chart(fig_importance,config={"responsive": True, "displaylogo": False}, key="importance_tab")
        st.dataframe(imp_df.head(20), width='stretch')
    else:
        if model_key == "mlp":
            st.warning("Feature importance is not applicable for Neural Network (MLP) models.")
        elif model_key == "lr":
            st.warning("Feature importance not available for this Logistic Regression configuration.")
        else:
            st.info("Feature importance not computed or unavailable.")

with tabs[3]:
    st.subheader("Risk vs Value — Quadrant")
    st.plotly_chart(fig_quadrant, config={"responsive": True, "displaylogo": False}, width='stretch', key="quadrant_tab")
    st.dataframe(seg_df, width='stretch')

with tabs[4]:
    st.subheader("CLV Distribution")
    st.plotly_chart(fig_clv, width='stretch', key="clv_tab")

with tabs[5]:
    st.subheader("Retention Curve (Tenure-based)")
    if fig_retention is not None:
        st.plotly_chart(fig_retention,config={"responsive": True, "displaylogo": False}, key="retention_tab")
    else:
        st.info("Tenure not provided — curve unavailable.")

with tabs[6]:
    st.subheader("Top At-Risk Customers")
    st.dataframe(top_risk_df, width='stretch')

with tabs[7]:
    st.subheader("Top Customers by Value")
    st.dataframe(top_value_df, width='stretch')

with tabs[8]:
    st.subheader("Recommended Actions")
    for r in generate_recommendations(metrics, portfolio, risk_threshold=risk_thr, value_threshold=value_thr):
        st.markdown(f"- {r}")

with tabs[9]:
    st.header("🖨 All Plots & Tables (Printable View)")
    st.markdown("Use your browser's **Print → Save as PDF** option to export this page.")
    st.subheader("Summary")
    st.table(summary_df)
    st.subheader("Churn Probability — Density")
    st.plotly_chart(fig_density,config={"responsive": True, "displaylogo": False}, key="density_all")
    st.dataframe(cm_df, width='stretch')
    if fig_importance is not None:
        st.subheader("Feature Importance")
        st.plotly_chart(fig_importance,config={"responsive": True, "displaylogo": False}, key="importance_all")
        st.dataframe(imp_df.head(20), width='stretch')
    st.subheader("Risk vs Value — Quadrant")
    st.plotly_chart(fig_quadrant, config={"responsive": True, "displaylogo": False}, key="quadrant_all")
    st.dataframe(seg_df, width='stretch')
    st.subheader("CLV Distribution")
    st.plotly_chart(fig_clv, config={"responsive": True, "displaylogo": False}, key="clv_all")
    if fig_retention is not None:
        st.subheader("Retention Curve (Tenure-based)")
        st.plotly_chart(fig_retention, config={"responsive": True, "displaylogo": False}, key="retention_all")
    st.subheader("Top At-Risk Customers")
    st.dataframe(top_risk_df, width='stretch')
    st.subheader("Top Customers by Value")
    st.dataframe(top_value_df, width='stretch')
    st.subheader("Recommendations")
    for r in generate_recommendations(metrics, portfolio, risk_threshold=risk_thr, value_threshold=value_thr):
        st.markdown(f"- {r}")
    st.markdown("---")
    st.caption("End of printable report — Retention Analytics Module.")

# ---------------------------------------------------------------------
# Future Steps & Premium Add-ons
# ---------------------------------------------------------------------
with tabs[10]:
    st.header("Future Steps & Premium Add-ons")
    st.markdown("""
    The **Retention • Churn & Value Analytics** module currently delivers the complete foundation for customer-level risk and value analysis, including:
    - Churn modeling using 4 machine learning algorithms (Logistic Regression, Random Forest, XGBoost, and MLP)
    - Customer Lifetime Value (CLV) estimation based on churn probability and revenue
    - 2×2 segmentation by risk and value
    - Interactive portfolio analysis and retention curve visualization

    ---
    ### Planned Extensions — *Advanced & Premium Add-ons*

    | Category | Enhancement | Description |
    |-----------|-------------|-------------|
    | **Model Optimization** | **Calibration (Platt / Isotonic)** | Improves churn probability realism and CLV accuracy through post-training calibration. |
    | **Model Robustness** | **Cross-Validation & Auto Tuning** | Adds K-Fold or Bayesian optimization for stable, reproducible model performance. |
    | **Explainability** | **SHAP / LIME Analysis** | Enables granular feature contribution analysis and visual explanations for business users. |
    | **Value Modeling** | **Discounted / Contractual CLV** | Extends the CLV formula with discounting and tenure sensitivity for long-term forecasts. |
    | **Strategy Simulation** | **Retention ROI Planner** | Simulates business impact of reducing churn or improving value by user-defined percentages. |
    | **Data Governance** | **Schema Validation & Drift Tracking** | Ensures robust use across datasets and time periods by validating inputs and detecting drift. |

    ---
    ### Availability
    These features are part of the **Enterprise / Premium build** of this module.
    
    To enable, extend, or commission the advanced analytics features, please reach out for integration or freelance collaboration.

    **Current Edition:** Core (Open) — fully functional for learning and demonstration purposes.
    """)

    st.info("You're currently using the Core Edition — optimized for learning, portfolio, and demonstration.")
