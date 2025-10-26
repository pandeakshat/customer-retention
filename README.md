# Retention • Churn & Value Analytics

A complete, modular **Customer Retention & Value Analytics** application built with **Streamlit**, designed to:
- Model **customer churn risk**
- Estimate **Customer Lifetime Value (CLV)**
- Segment portfolios by **Risk–Value**
- Visualize **retention insights**
- Serve as both a **learning project** and **professional analytics showcase**

---

## 🚀 Overview

This module is part of a broader **Customer Intelligence Hub** concept and demonstrates:
- End-to-end machine learning workflows (data → model → business insight)
- Integration of business KPIs such as CLV, concentration, and retention rate
- Interactive analytics powered by Streamlit and Plotly

It is structured to function as:
- A **learning lab** for churn and CLV analytics  
- A **portfolio project** to demonstrate ML + business understanding  
- A **freelance-ready base** for customized enterprise retention analytics  

---

## 🧩 Features

| Category | Capability | Description |
|-----------|-------------|-------------|
| **Modeling** | Logistic Regression, Random Forest, XGBoost, MLP | 4 core models with parameter sliders and training controls |
| **Data Input** | CSV Upload or Sample Dataset | Works with any customer dataset containing a binary churn indicator |
| **Evaluation** | AUC, Precision, Recall, F1, Confusion Matrix | Provides instant model evaluation metrics |
| **Customer Value** | CLV Estimation | Calculates expected customer value from churn risk, ARPU, and margin |
| **Segmentation** | Risk–Value Quadrant | 2×2 segmentation: High/Low Risk × High/Low Value |
| **Visualizations** | Interactive Plotly Charts | Churn density, feature importance, CLV distribution, retention curve |
| **Portfolio Metrics** | Total Value, Concentration, Retention | Business summaries for portfolio-level insights |
| **Report Mode** | Printable “All Plots & Tables” Tab | Exports full report via browser Print → PDF |
| **Modes** | Pre-Trained / Retrain Toggle | Load pre-saved models or retrain on new datasets |

---

## 🧠 Learning Objectives

1. Understand the **drivers of customer churn** and how to model them.
2. Build and interpret **machine learning pipelines** using scikit-learn and XGBoost.
3. Learn to estimate **Customer Lifetime Value (CLV)** from churn probabilities.
4. Translate model outputs into **business-level retention and value insights**.

---

## 📂 Project Structure

```
customer-retention/
│
├── app.py                     # Streamlit app (interactive interface)
├── train.py                   # Offline model training script
│
├── utils/
│   ├── load.py                # Data loading, cleaning, validation
│   ├── retain.py              # ML modeling, CLV estimation, segmentation
│   └── report.py              # Plots, metrics tables, and recommendations
│
├── data/
│   └── sample.csv             # Sample Telco churn dataset
│
├── models/                    # Saved models (.pkl)
├── outputs/                   # Predictions, metrics, portfolio CSVs
└── README.md                  # Project documentation
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run model training (optional)
```bash
python train.py
```

### 3. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

You can upload your own dataset or use the built-in `data/sample.csv`.

---

## 🧮 Example Outputs

- **Model Summary:** AUC, Precision, Recall, F1  
- **Feature Importance:** Key churn drivers ranked by influence  
- **Risk–Value Quadrant:** Strategic segmentation for targeting  
- **CLV Distribution:** Customer value concentration insights  
- **Recommendations:** Automated, data-driven retention strategies  

---

## 🔮 Future Steps & Premium Add-ons

| Category | Enhancement | Description |
|-----------|-------------|-------------|
| 🎯 Model Optimization | Calibration (Platt / Isotonic) | Improves churn probability realism and CLV accuracy |
| 🔁 Model Robustness | Cross-Validation & Auto Tuning | Enables reproducible and optimized model performance |
| 💡 Explainability | SHAP / LIME Analysis | Adds visual interpretability for decision support |
| 💸 Value Modeling | Discounted / Contractual CLV | Integrates discounting and tenure sensitivity |
| 📈 Simulation | Retention ROI Planner | Projects financial impact of churn reduction |
| 🧮 Data Governance | Schema Validation & Drift Tracking | Enables robust reuse and continuous monitoring |

These are part of the **Premium / Enterprise Edition**, available for professional engagements.

---

## 🧾 Licensing & Use

This **Core Edition** is open for:
- Educational use 🧑‍🎓  
- Personal learning & portfolio demonstration 🧠  

Advanced / enterprise modules are available under a **freelance or consulting license**.

> Built by [Your Name] — *Retention Analytics Module (Core Edition)*  
> For advanced implementations or collaborations, please get in touch.

---

## 🧭 Next Steps

- [ ] Add calibration and cross-validation (advanced modeling)  
- [ ] Implement discounted CLV model  
- [ ] Integrate SHAP interpretability  
- [ ] Build ROI simulation planner  

---

## 📣 Contact

Interested in extending this module or collaborating on retention analytics projects?  
📧 **[Your Email or Portfolio Link]**

---

### 💡 Summary

This project bridges **machine learning, business analytics, and customer economics** into a single Streamlit application.  
It demonstrates **technical skill, interpretability, and domain understanding** — a full retention analytics lifecycle.
