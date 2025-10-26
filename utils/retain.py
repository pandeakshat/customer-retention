"""
retain.py — Core analytical methods (generic, 4-model suite + CLV + metrics)

Models supported via `model_key`:
  - 'lr'  : Logistic Regression (param: C)
  - 'rf'  : Random Forest (params: n_estimators, max_depth)
  - 'xgb' : XGBoost (params: n_estimators, learning_rate, max_depth)
  - 'mlp' : Neural Net (params: hidden_units, hidden_layers, learning_rate_init)

Includes:
- consistent binary label mapping (Yes/No → 1/0)
- preprocessing (impute, scale, OHE)
- probability scoring & metrics
- feature importances (RF, XGB, LR)
- CLV heuristic + portfolio indicators
- 2×2 segmentation
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier as _SK_MLP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

SEED = 42

# Optional XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

try:
    MLPClassifier = _SK_MLP
except Exception:
    MLPClassifier = None


# ---------------------------------------------------------------------
# Target Handling
# ---------------------------------------------------------------------
def split_features_target(df: pd.DataFrame, target_col: str, positive_label: Optional[object] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataset into X, y with consistent binary mapping:
    - Yes/Y/True/1 → 1
    - No/N/False/0 → 0
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col].astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}

    if positive_label is not None:
        pos = str(positive_label).strip().lower()
        y = (y_raw == pos).astype(int)
    else:
        y = y_raw.map(mapping)
        if y.isna().any():
            classes = y_raw.dropna().unique().tolist()
            if len(classes) == 2:
                # Choose minority class as positive by default
                counts = y_raw.value_counts()
                pos_label = counts.index[-1]
                y = (y_raw == pos_label).astype(int)
            else:
                raise ValueError("Target encoding failed — ensure binary target or specify positive_label.")

    X = df.drop(columns=[target_col])
    return X, y


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    blacklist = {"customerid", "customer_id", "customer_id_hash", "id", "uuid", "guid"}
    categorical_cols = [c for c in categorical_cols if c.lower() not in blacklist]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# ---------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------
def build_model_pipeline(X: pd.DataFrame, model_key: str, params: Dict[str, object]) -> Tuple[Pipeline, List[str], List[str]]:
    num_cols, cat_cols = infer_feature_types(X)
    pre = build_preprocessor(num_cols, cat_cols)

    if model_key == "lr":
        clf = LogisticRegression(random_state=SEED, max_iter=1000, C=float(params.get("C", 1.0)))
    elif model_key == "rf":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 12)),
            random_state=SEED,
            n_jobs=-1,
        )
    elif model_key == "xgb":
        if XGBClassifier is None:
            raise ImportError("XGBoost not installed. Run `pip install xgboost`.")
        clf = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
    elif model_key == "mlp":
        if MLPClassifier is None:
            raise ImportError("MLPClassifier not available. Ensure scikit-learn is installed.")
        hidden = int(params.get("hidden_units", 64))
        layers = int(params.get("hidden_layers", 2))
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden,) * layers,
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            random_state=SEED,
            max_iter=300,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unsupported model_key '{model_key}'.")

    return Pipeline(steps=[("pre", pre), ("clf", clf)]), num_cols, cat_cols


def fit_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe.fit(X, y)
    return pipe


def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------
# Metrics & Importance
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC": float(auc),
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def get_feature_importance_df(model, feature_names):
    """
    Extract feature importance for RandomForest, XGBoost, LogisticRegression.
    """
    import numpy as np
    import pandas as pd

    model_name = model.__class__.__name__.lower()
    imp = None

    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).flatten()
        elif "xgb" in model_name and hasattr(model, "get_booster"):
            booster = model.get_booster()
            score_dict = booster.get_score(importance_type="gain")
            imp = np.array([score_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])

        if imp is None or len(imp) != len(feature_names):
            return None

        df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        df["Feature"] = df["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)
        return df.sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception:
        return None


# ---------------------------------------------------------------------
# CLV & Segmentation
# ---------------------------------------------------------------------
def estimate_clv(df: pd.DataFrame, churn_prob_col: str, arpu_col: Optional[str] = None,
                 default_arpu: float = 50.0, margin: float = 0.6, max_months: int = 60) -> pd.Series:
    if churn_prob_col not in df.columns:
        raise KeyError(f"Missing churn probability column: '{churn_prob_col}'")
    p = df[churn_prob_col].clip(1e-6, 1 - 1e-6)
    expected_months = (1.0 / p).clip(1, max_months)
    if arpu_col and arpu_col in df.columns:
        arpu = pd.to_numeric(df[arpu_col], errors="coerce").fillna(default_arpu)
    else:
        arpu = pd.Series(np.full(len(df), default_arpu), index=df.index)
    return arpu * float(margin) * expected_months


def clv_portfolio_metrics(df: pd.DataFrame, clv_col: str, churn_prob_col: str, top_share: float = 0.10) -> Dict[str, float]:
    if clv_col not in df.columns or churn_prob_col not in df.columns:
        return {"total_expected_value": np.nan, "value_concentration_top_share": np.nan,
                "high_value_retention_rate": np.nan, "avg_expected_months": np.nan}
    total_ev = float(df[clv_col].sum())
    n_top = max(1, int(np.floor(len(df) * top_share)))
    top_df = df.nlargest(n_top, clv_col)
    concentration = float(top_df[clv_col].sum() / (total_ev + 1e-9))
    high_value_retention = float(1.0 - top_df[churn_prob_col].mean())
    avg_months = float((1.0 / df[churn_prob_col].clip(1e-6, 1 - 1e-6)).mean())
    return {
        "total_expected_value": total_ev,
        "value_concentration_top_share": concentration,
        "high_value_retention_rate": high_value_retention,
        "avg_expected_months": avg_months,
    }


def segment_customers(df: pd.DataFrame, prob_col: str, clv_col: str,
                      risk_thr: Optional[float] = None, value_thr: Optional[float] = None) -> pd.Series:
    if risk_thr is None:
        risk_thr = float(df[prob_col].median())
    if value_thr is None:
        value_thr = float(df[clv_col].median())
    def label(row):
        risk = "High Risk" if row[prob_col] >= risk_thr else "Low Risk"
        value = "High Value" if row[clv_col] >= value_thr else "Low Value"
        return f"{risk} • {value}"
    return df.apply(label, axis=1)
