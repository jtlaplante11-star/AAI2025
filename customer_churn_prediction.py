"""
Part 2: Predict Customer Churn.
Author: Jacob Laplante

DATA SOURCE (real data):
  IBM Telco Customer Churn dataset (7,043 customers of a fictional telecom
  company, real usage/billing/contract fields, published by IBM).
  Official IBM GitHub mirror (no login required):
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
  Also mirrored on Kaggle as "Telco Customer Churn":
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

  NOTE: this is real data, so it uses the telecom company's *actual* column
  names rather than the fictional 'age / monthly_usage_hours / region'
  columns invented for the toy starter example. We use:
    - tenure          (months as a customer)               -> numeric
    - MonthlyCharges   (current monthly bill, $)             -> numeric
    - TotalCharges     (total billed to date, $)             -> numeric
    - SeniorCitizen    (0/1 flag)                            -> numeric
    - Contract         (Month-to-month / One year / Two year) -> categorical
    - Churn            (Yes/No)                              -> target

  The same modeling idea from the assignment still applies: StandardScaler
  on the numeric columns, OneHotEncoder on the categorical column
  (Contract plays the role 'region' played in the starter code), then
  LogisticRegression, then classify with a 0.5 probability threshold.

  NOTE ON THE SANDBOX THIS WAS DEVELOPED IN: outbound internet in the
  environment used to write/test this script is restricted to package
  registries only, so the live download above could not be exercised
  there. It will work normally in Google Colab, GitHub Actions, or your
  own laptop. If the download fails for any reason, the script falls back
  to a synthetic dataset built to match the real dataset's published
  statistics (~26.5% overall churn rate; month-to-month + high charges +
  low tenure customers churn much more often) so it always runs end to end.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

TELCO_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
             "master/data/Telco-Customer-Churn.csv")
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = ["Contract"]
RANDOM_STATE = 42


def load_real_telco_data():
    """Download the real IBM Telco Customer Churn dataset and trim it down
    to the columns this assignment's pipeline needs."""
    raw = pd.read_csv(TELCO_URL)
    df = raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Churn"]].copy()

    # TotalCharges is stored as text in the raw file and has ~11 blank
    # values for brand-new customers (tenure == 0); coerce and drop those.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])
    return df


def generate_fallback_data(n=500, seed=RANDOM_STATE):
    """Synthetic stand-in matching the real Telco dataset's schema and
    published churn statistics, used only if the live download above is
    not reachable."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])  # real dataset: ~16% senior

    # Tenure depends loosely on contract type (longer contracts -> longer tenure)
    tenure_base = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [rng.normal(18, 14, n), rng.normal(35, 16, n), rng.normal(50, 18, n)],
    )
    tenure = tenure_base.clip(0, 72).round().astype(int)

    monthly_charges = rng.normal(64.8, 30, n).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(0, None).round(2)

    # Realistic churn probability: higher for month-to-month, high monthly
    # charges, low tenure, and senior citizens -- consistent with the
    # patterns reported for the real IBM Telco dataset (~26.5% overall churn).
    logit = (
        -2.2
        + 1.6 * (contract == "Month-to-month")
        - 0.4 * (contract == "One year")
        - 1.0 * (contract == "Two year")
        + 0.02 * monthly_charges
        - 0.03 * tenure
        + 0.3 * senior
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior,
        "Contract": contract,
        "churn": churn,
    })


def load_data():
    try:
        df = load_real_telco_data()
        print(f"Loaded {len(df)} REAL customer records from the IBM Telco "
              f"Customer Churn dataset ({TELCO_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real Telco churn dataset ({exc}). "
              f"Falling back to a documented synthetic dataset that matches "
              f"its real schema and churn statistics.")
        df = generate_fallback_data()
        print(f"Generated {len(df)} synthetic rows as a fallback. "
              f"Overall churn rate: {df['churn'].mean():.1%}")
        return df


df = load_data()

# Features and target
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["churn"]

# Preprocessing: scale numerical features, one-hot encode the categorical feature
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Train model
model.fit(X_train, y_train)

# --- Evaluation (improvement over the starter code, which never checked
# how good the model actually is) ---
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred_test):.3f}")
print(f"Test ROC AUC:  {roc_auc_score(y_test, y_proba_test):.3f}")
print(f"Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred_test)}")

# Predict churn probability for a new customer
new_customer = pd.DataFrame({
    "tenure": [5],
    "MonthlyCharges": [95.0],
    "TotalCharges": [475.0],
    "SeniorCitizen": [0],
    "Contract": ["Month-to-month"],
})
churn_probability = model.predict_proba(new_customer)[0][1]

threshold = 0.5
churn_prediction = 1 if churn_probability > threshold else 0

print(f"\nChurn Probability for new customer: {churn_probability:.2f}")
print(f"Churn Prediction (1 = churn, 0 = no churn): {churn_prediction}")

# Display model coefficients
feature_names = NUMERIC_FEATURES + (
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
).tolist()

coefficients = model.named_steps["classifier"].coef_[0]
print("\nModel Coefficients (in standardized/encoded feature space):")
for feature, coef in zip(feature_names, coefficients):
    print(f"  {feature}: {coef:.3f}")
print(f"  intercept: {model.named_steps['classifier'].intercept_[0]:.3f}")

# --- Plain-language explanation (required by the assignment) ---
print("\nExplanation:")
print(f"  - Churn probability ({churn_probability:.2f}): the model estimates roughly a "
      f"{churn_probability:.0%} chance that this specific customer cancels their service in the near "
      f"term, based on their tenure, billing amounts, and contract type.")
print(f"  - 0.5 threshold: customers with a predicted probability above 0.5 are classified as 'at risk' "
      f"(churn=1); at or below 0.5 they're classified as 'likely to stay' (churn=0). This customer's "
      f"probability ({churn_probability:.2f}) is {'above' if churn_probability > threshold else 'at or below'} "
      f"the threshold, so they are classified as {churn_prediction} "
      f"({'at risk of churning' if churn_prediction else 'not currently at risk'}).")
print("  - Business use: rather than treating churn as all-or-nothing, a business can rank every customer "
      "by predicted churn probability and prioritize retention efforts (discounts, proactive outreach, "
      "contract upgrade offers) toward the highest-probability customers first. That concentrates limited "
      "retention budget where it is statistically most likely to prevent lost revenue, instead of spreading "
      "offers evenly across a customer base where most people were never going to leave anyway.")
