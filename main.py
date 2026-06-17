import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
# Using synthetic data since no CSV is provided.
# Replace this block with: df = pd.read_csv("your_file.csv")

def generate_dataset(n=7000, seed=42):
    np.random.seed(seed)
    tenure       = np.random.randint(1, 73, n)
    monthly      = np.round(np.random.uniform(20, 120, n), 2)
    gender       = np.random.choice(["Male", "Female"], n)
    senior       = np.random.choice([0, 1], n, p=[0.84, 0.16])
    contract     = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    internet     = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment      = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    paperless    = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n)
    total_charges = np.round(tenure * monthly * np.random.uniform(0.9, 1.1, n), 2)

    # Churn probability influenced by contract type and tenure
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        - 0.002 * tenure
        + 0.003 * (monthly - 65)
        + 0.10 * (internet == "Fiber optic")
        + 0.08 * senior
    )
    churn_prob = np.clip(churn_prob, 0.03, 0.85)
    churn = np.where(np.random.uniform(0, 1, n) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID":    [f"C{str(i).zfill(5)}" for i in range(n)],
        "gender":        gender,
        "SeniorCitizen": senior,
        "tenure":        tenure,
        "Contract":      contract,
        "InternetService": internet,
        "TechSupport":   tech_support,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "MonthlyCharges":  monthly,
        "TotalCharges":    total_charges,
        "Churn":           churn,
    })

    # Inject some missing values and duplicates for realism
    df.loc[np.random.choice(df.index, 80, replace=False), "TotalCharges"] = np.nan
    df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df):
    print(f"Raw shape: {df.shape}")

    df.drop_duplicates(inplace=True)
    print(f"After dropping duplicates: {df.shape}")

    # TotalCharges can come in as object from real CSVs
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    print(f"Filled {missing} missing TotalCharges with median")

    # SeniorCitizen is 0/1 in the raw data – keep it numeric
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Drop the ID column – not useful for modelling
    df.drop(columns=["customerID"], inplace=True)

    print(f"Clean shape: {df.shape}\n")
    return df


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def run_eda(df):
    print("=== EDA SUMMARY ===")
    churn_rate = df["Churn"].value_counts(normalize=True) * 100
    print(f"Churn rate: {churn_rate['Yes']:.1f}%\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Churn – Exploratory Analysis", fontsize=15, weight="bold")

    # Churn distribution
    ax = axes[0, 0]
    df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Churn Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No", "Yes"], rotation=0)

    # Churn by gender
    ax = axes[0, 1]
    gender_churn = df.groupby("gender")["Churn"].value_counts(normalize=True).unstack() * 100
    gender_churn.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Churn Rate by Gender")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Churn")

    # Churn by contract type
    ax = axes[0, 2]
    contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack() * 100
    contract_churn.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="Churn")

    # Monthly charges distribution
    ax = axes[1, 0]
    df[df["Churn"] == "Yes"]["MonthlyCharges"].hist(ax=ax, bins=30, color="#e74c3c", alpha=0.6, label="Churned")
    df[df["Churn"] == "No"]["MonthlyCharges"].hist(ax=ax, bins=30, color="#2ecc71", alpha=0.6, label="Retained")
    ax.set_title("Monthly Charges Distribution")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    ax.legend()

    # Tenure distribution
    ax = axes[1, 1]
    df[df["Churn"] == "Yes"]["tenure"].hist(ax=ax, bins=30, color="#e74c3c", alpha=0.6, label="Churned")
    df[df["Churn"] == "No"]["tenure"].hist(ax=ax, bins=30, color="#2ecc71", alpha=0.6, label="Retained")
    ax.set_title("Tenure Distribution")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend()

    # Churn by Internet Service
    ax = axes[1, 2]
    internet_churn = df.groupby("InternetService")["Churn"].value_counts(normalize=True).unstack() * 100
    internet_churn.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Churn Rate by Internet Service")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="Churn")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/eda_plots.png")


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    le = LabelEncoder()

    # Binary columns – simple yes/no mapping
    binary_cols = ["PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # gender
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # TechSupport – treat "No internet service" as No
    df["TechSupport"] = df["TechSupport"].replace("No internet service", "No")
    df["TechSupport"] = df["TechSupport"].map({"Yes": 1, "No": 0})

    # Multi-class categoricals – label encode for RF, good enough here
    multi_cat = ["Contract", "InternetService", "PaymentMethod"]
    for col in multi_cat:
        df[col] = le.fit_transform(df[col])

    # Safety net: fill any remaining NaNs introduced during encoding
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


# ─────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────────

def plot_correlation(df):
    plt.figure(figsize=(11, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, annot_kws={"size": 8})
    plt.title("Feature Correlation Heatmap", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/correlation_heatmap.png")


# ─────────────────────────────────────────────
# 6. MODEL BUILDING & EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name, y_test, y_pred):
    print(f"\n── {name} ──")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    return {
        "model": name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "cm":        confusion_matrix(y_test, y_pred),
    }


def plot_confusion_matrix(cm, model_name, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


def build_models(df):
    feature_cols = [c for c in df.columns if c != "Churn"]
    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_results = evaluate_model("Logistic Regression", y_test, lr.predict(X_test_sc))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_results = evaluate_model("Random Forest", y_test, rf.predict(X_test))

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(lr_results["cm"], "Logistic Regression", axes[0])
    plot_confusion_matrix(rf_results["cm"], "Random Forest", axes[1])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: output/confusion_matrices.png")

    # Feature importance from Random Forest
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="#3498db", edgecolor="white")
    plt.title("Feature Importance – Random Forest", fontsize=13, weight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/feature_importance.png")

    return lr_results, rf_results, importances


# ─────────────────────────────────────────────
# 7. BUSINESS INSIGHTS
# ─────────────────────────────────────────────

def print_insights(df_clean, importances):
    print("\n" + "=" * 50)
    print("BUSINESS INSIGHTS")
    print("=" * 50)

    top3 = importances.sort_values(ascending=False).head(3).index.tolist()
    print(f"\nTop 3 churn drivers: {', '.join(top3)}")

    # Month-to-month churn rate
    mtm_churn = df_clean[df_clean["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True)
    if "Yes" in mtm_churn:
        print(f"Month-to-month churn rate : {mtm_churn['Yes']*100:.1f}%")

    avg_tenure_churned  = df_clean[df_clean["Churn"] == "Yes"]["tenure"].mean()
    avg_tenure_retained = df_clean[df_clean["Churn"] == "No"]["tenure"].mean()
    print(f"Avg tenure – churned  : {avg_tenure_churned:.1f} months")
    print(f"Avg tenure – retained : {avg_tenure_retained:.1f} months")

    print("\nRetention Recommendations:")
    print("  1. Target month-to-month customers with discounted annual plans.")
    print("  2. Flag new customers (tenure < 6 months) for proactive outreach.")
    print("  3. Investigate Fiber Optic churn – possible service quality issue.")
    print("  4. Senior citizens show elevated churn; tailor support programs.")
    print("  5. Customers on electronic check payment tend to churn more –")
    print("     incentivise auto-pay enrollment.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("TELECOM CUSTOMER CHURN ANALYSIS")
    print("=" * 50 + "\n")

    # Load (or generate) data
    df_raw = generate_dataset()
    # df_raw = pd.read_csv("telco_churn.csv")   # ← swap in for real data

    df_clean = clean_data(df_raw.copy())

    run_eda(df_clean)

    plot_correlation(engineer_features(df_clean))

    df_encoded = engineer_features(df_clean)

    print("\n=== MODEL RESULTS ===")
    lr_res, rf_res, importances = build_models(df_encoded)

    print_insights(df_clean, importances)

    print("\n✓ All outputs saved to the 'output/' folder.")
