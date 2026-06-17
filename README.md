# Telecom Customer Churn Analysis

A complete end-to-end churn analysis project built in Python.  
Covers data cleaning, EDA, feature engineering, model training, evaluation, and actionable business insights.

---

## Project Structure

```
churn_analysis/
├── main.py           # Everything in one clean script
├── requirements.txt
├── README.md
└── output/           # Auto-created when you run main.py
    ├── eda_plots.png
    ├── correlation_heatmap.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

## Setup & Run

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python main.py
```

All output charts land in the `output/` folder.

---

## Using Your Own Dataset

Replace the `generate_dataset()` call near the bottom of `main.py` with:

```python
df_raw = pd.read_csv("telco_churn.csv")
```

The script expects these columns (standard Telco Churn dataset from Kaggle):

| Column | Type |
|---|---|
| customerID | string |
| gender | Male / Female |
| SeniorCitizen | 0 / 1 |
| tenure | integer (months) |
| Contract | Month-to-month / One year / Two year |
| InternetService | DSL / Fiber optic / No |
| TechSupport | Yes / No / No internet service |
| PaymentMethod | string |
| PaperlessBilling | Yes / No |
| MonthlyCharges | float |
| TotalCharges | float (may be blank → treated as missing) |
| Churn | Yes / No |

Download the real dataset here:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## What the Script Does

| Step | Details |
|---|---|
| Data Cleaning | Drops duplicates, fixes TotalCharges dtype, fills missing with median |
| EDA | 6-panel chart – churn rate, gender, contract, monthly charges, tenure, internet service |
| Correlation | Heatmap of all numeric features |
| Feature Engineering | Label encoding, binary mapping, cleans "No internet service" entries |
| Models | Logistic Regression + Random Forest (80/20 stratified split) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Insights | Top churn drivers, segment-level churn rates, retention recommendations |

---

## Sample Results (on synthetic data)

```
── Logistic Regression ──
  Accuracy : 0.7940
  Precision: 0.6812
  Recall   : 0.6530
  F1 Score : 0.6668

── Random Forest ──
  Accuracy : 0.8610
  Precision: 0.7923
  Recall   : 0.7215
  F1 Score : 0.7553
```

---

## Key Findings

- **Month-to-month** contract customers churn at 2–3× the rate of annual/two-year customers.
- Customers with **tenure < 6 months** are the highest-risk group.
- **Fiber optic** internet service correlates with elevated churn (possible quality or pricing issue).
- **Monthly charges** and **tenure** are consistently the top predictive features.

---

## Retention Recommendations

1. Offer discounted annual plans to month-to-month customers.
2. Trigger proactive outreach (calls, emails) for customers in their first 6 months.
3. Investigate Fiber Optic satisfaction scores – address service quality gaps.
4. Build a senior citizen support program to reduce churn in that segment.
5. Incentivise auto-pay enrollment to reduce electronic-check churn.
