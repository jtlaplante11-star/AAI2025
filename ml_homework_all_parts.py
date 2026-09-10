"""
AAI 2025 - Machine Learning Homework (all parts combined into one file)
Author: Jacob Laplante

Part 1          - House price prediction (Linear Regression)
Part 2          - Customer churn prediction (Logistic Regression)
Part 3          - Customer segmentation (K-Means)
Extra Credit    - Housing demand forecasting (Linear Regression + seasonality)

Each part is self-contained in its own function (own data loader, own model,
own printed results) so variable names never collide between parts, even
though everything lives in this single file. Run the whole thing with:

    python ml_homework_all_parts.py

Each part tries to download its REAL public dataset first. If that fails
(no internet, host down, etc.) it automatically falls back to a documented
synthetic dataset built to match that real dataset's published statistics,
so the script always runs end to end either way -- the printed output
tells you which path was used for each part.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# PART 1: HOUSE PRICE PREDICTION
#
# DATA SOURCE (real data): Ames Housing Dataset (De Cock, D., 2011, "Ames,
# Iowa: Alternative to the Boston Housing Data as an End of Semester
# Regression Project", Journal of Statistics Education, 19(3)).
#   https://jse.amstat.org/v19n3/decock/AmesHousing.txt
#   Data dictionary: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt
# 2,930 real home sales in Ames, IA (2006-2010). We use:
#   'Gr Liv Area' -> square_footage, 'Neighborhood' -> location (top 6 most
#   common neighborhoods only, for a clean small category set, mirroring
#   the toy 'Downtown/Suburb/Rural' example in the original starter code),
#   'SalePrice' -> price (target).
# ======================================================================

AMES_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"


def load_real_ames_data(top_n=6):
    raw = pd.read_csv(AMES_URL, sep="\t")
    df = raw[["Gr Liv Area", "Neighborhood", "SalePrice"]].rename(
        columns={"Gr Liv Area": "square_footage", "Neighborhood": "location", "SalePrice": "price"}
    ).dropna()
    top_locations = df["location"].value_counts().nlargest(top_n).index
    return df[df["location"].isin(top_locations)].reset_index(drop=True)


def generate_fallback_house_data(n=300, seed=RANDOM_STATE):
    """Synthetic stand-in matching Ames Housing's published summary stats
    (mean SalePrice ~$180,796, mean Gr Liv Area ~1,515 sqft, ~$100-120
    marginal price per sqft), used only if the live download is unreachable."""
    rng = np.random.default_rng(seed)
    neighborhoods = {
        "NAmes": (145000, 95), "CollgCr": (200000, 110), "OldTown": (125000, 85),
        "Edwards": (130000, 90), "Somerst": (225000, 120), "Gilbert": (190000, 105),
    }
    names = list(neighborhoods.keys())
    locations = rng.choice(names, size=n)
    square_footage = rng.normal(1515, 500, size=n).clip(600, 4200).round().astype(int)
    base_price = np.array([neighborhoods[loc][0] for loc in locations])
    per_sqft = np.array([neighborhoods[loc][1] for loc in locations])
    noise = rng.normal(0, 18000, size=n)
    price = (base_price + per_sqft * square_footage + noise).clip(60000, 600000).round(-2)
    return pd.DataFrame({"square_footage": square_footage, "location": locations, "price": price})


def load_house_data():
    try:
        df = load_real_ames_data()
        print(f"Loaded {len(df)} REAL home sales from the Ames Housing dataset ({AMES_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real Ames Housing dataset ({exc}). "
              f"Falling back to a documented synthetic dataset that matches its real summary statistics.")
        df = generate_fallback_house_data()
        print(f"Generated {len(df)} synthetic rows as a fallback.")
        return df


def run_part1_house_prices():
    section("PART 1: HOUSE PRICE PREDICTION")
    df = load_house_data()

    X = df[["square_footage", "location"]]
    y = df["price"]

    preprocessor = ColumnTransformer(
        transformers=[("location", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["location"])],
        remainder="passthrough",
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    print(f"\nTest R^2:  {r2_score(y_test, y_pred_test):.3f}")
    print(f"Test MAE:  ${mean_absolute_error(y_test, y_pred_test):,.0f}")

    # NOTE: the assignment's original example asks for a prediction in
    # 'Downtown'. Since we upgraded to the real Ames Housing dataset, there
    # is no neighborhood literally named 'Downtown' -- so this predicts for
    # the dataset's most common neighborhood as the representative example.
    example_location = df["location"].value_counts().idxmax()
    new_house = pd.DataFrame({"square_footage": [2000], "location": [example_location]})
    predicted_price = model.predict(new_house)
    print(f"\nPredicted price for a 2000 sq ft house in {example_location} "
          f"(this dataset's real-data equivalent of the starter code's 'Downtown' example): "
          f"${predicted_price[0]:,.2f}")

    feature_names = (
        model.named_steps["preprocessor"].named_transformers_["location"].get_feature_names_out(["location"])
    ).tolist() + ["square_footage"]
    coefficients = model.named_steps["regressor"].coef_
    print("\nModel Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"  {feature}: {coef:,.2f}")
    print(f"  intercept: {model.named_steps['regressor'].intercept_:,.2f}")

    # --- Plain-language explanation (required by the assignment) ---
    sqft_coef = coefficients[-1]
    location_coefs = {name.split("_", 1)[-1]: coef for name, coef in zip(feature_names[:-1], coefficients[:-1])}
    priciest_loc, priciest_val = max(location_coefs.items(), key=lambda kv: kv[1])
    cheapest_loc, cheapest_val = min(location_coefs.items(), key=lambda kv: kv[1])
    print("\nExplanation:")
    print(f"  - Square footage coefficient (${sqft_coef:,.2f}/sqft): holding the neighborhood fixed, each "
          f"additional square foot of living area is associated with about ${sqft_coef:,.2f} more in predicted "
          f"sale price.")
    print(f"  - Location effect: each neighborhood's coefficient shows how much more (positive) or less "
          f"(negative) homes there sell for compared to other neighborhoods, at the same square footage. "
          f"Here, '{priciest_loc}' carries the largest premium (${priciest_val:,.0f}), while '{cheapest_loc}' "
          f"carries the largest discount (${cheapest_val:,.0f}).")


# ======================================================================
# PART 2: CUSTOMER CHURN PREDICTION
#
# DATA SOURCE (real data): IBM Telco Customer Churn dataset (7,043 real
# customers). Official IBM GitHub mirror:
#   https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# Also on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Because this is real data, it uses the telecom company's actual column
# names (tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Contract)
# rather than the fictional 'age/region' columns from the toy starter
# example -- Contract plays the same OneHotEncoder-demo role 'region' did.
# ======================================================================

TELCO_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
             "master/data/Telco-Customer-Churn.csv")
CHURN_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CHURN_CATEGORICAL_FEATURES = ["Contract"]


def load_real_telco_data():
    raw = pd.read_csv(TELCO_URL)
    df = raw[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES + ["Churn"]].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["Churn"])


def generate_fallback_churn_data(n=500, seed=RANDOM_STATE):
    """Synthetic stand-in matching the real Telco dataset's schema and
    published churn statistics (~26.5% overall churn rate; month-to-month
    + high charges + low tenure customers churn much more often)."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    tenure_base = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [rng.normal(18, 14, n), rng.normal(35, 16, n), rng.normal(50, 18, n)],
    )
    tenure = tenure_base.clip(0, 72).round().astype(int)
    monthly_charges = rng.normal(64.8, 30, n).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(0, None).round(2)
    logit = (
        -2.2 + 1.6 * (contract == "Month-to-month") - 0.4 * (contract == "One year")
        - 1.0 * (contract == "Two year") + 0.02 * monthly_charges - 0.03 * tenure + 0.3 * senior
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "SeniorCitizen": senior, "Contract": contract, "churn": churn,
    })


def load_churn_data():
    try:
        df = load_real_telco_data()
        print(f"Loaded {len(df)} REAL customer records from the IBM Telco Customer Churn dataset ({TELCO_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real Telco churn dataset ({exc}). "
              f"Falling back to a documented synthetic dataset that matches its real schema and churn statistics.")
        df = generate_fallback_churn_data()
        print(f"Generated {len(df)} synthetic rows as a fallback. Overall churn rate: {df['churn'].mean():.1%}")
        return df


def run_part2_customer_churn():
    section("PART 2: CUSTOMER CHURN PREDICTION")
    df = load_churn_data()

    X = df[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES]
    y = df["churn"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), CHURN_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CHURN_CATEGORICAL_FEATURES),
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred_test):.3f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, y_proba_test):.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, y_pred_test)}")

    new_customer = pd.DataFrame({
        "tenure": [5], "MonthlyCharges": [95.0], "TotalCharges": [475.0],
        "SeniorCitizen": [0], "Contract": ["Month-to-month"],
    })
    churn_probability = model.predict_proba(new_customer)[0][1]
    threshold = 0.5
    churn_prediction = 1 if churn_probability > threshold else 0
    print(f"\nChurn Probability for new customer: {churn_probability:.2f}")
    print(f"Churn Prediction (1 = churn, 0 = no churn): {churn_prediction}")

    feature_names = CHURN_NUMERIC_FEATURES + (
        model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(CHURN_CATEGORICAL_FEATURES)
    ).tolist()
    coefficients = model.named_steps["classifier"].coef_[0]
    print("\nModel Coefficients (in standardized/encoded feature space):")
    for feature, coef in zip(feature_names, coefficients):
        print(f"  {feature}: {coef:.3f}")
    print(f"  intercept: {model.named_steps['classifier'].intercept_[0]:.3f}")

    # --- Plain-language explanation (required by the assignment) ---
    print("\nExplanation:")
    print(f"  - Churn probability ({churn_probability:.2f}): the model estimates roughly a "
          f"{churn_probability:.0%} chance this specific customer cancels their service in the near term, "
          f"based on their tenure, billing amounts, and contract type.")
    print(f"  - 0.5 threshold: probabilities above 0.5 are classified as 'at risk' (churn=1); at or below "
          f"0.5 as 'likely to stay' (churn=0). This customer ({churn_probability:.2f}) is classified as "
          f"{churn_prediction} ({'at risk of churning' if churn_prediction else 'not currently at risk'}).")
    print("  - Business use: a business can rank all customers by predicted churn probability and prioritize "
          "retention offers (discounts, proactive outreach, contract upgrades) toward the highest-probability "
          "customers first, concentrating limited retention budget where it's most likely to prevent lost "
          "revenue.")


# ======================================================================
# PART 3: CUSTOMER SEGMENTATION (K-MEANS)
#
# DATA SOURCE (real data): Mall Customer Segmentation Data (200 real mall
# customers). Kaggle: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# No-login CSV mirror: https://raw.githubusercontent.com/erkansirin78/datasets/master/Mall_Customers.csv
# Columns are located by keyword (not exact name) because different
# mirrors spell headers slightly differently (e.g. 'Genre' vs 'Gender').
# ======================================================================

MALL_URL = "https://raw.githubusercontent.com/erkansirin78/datasets/master/Mall_Customers.csv"
SEGMENTATION_FEATURES = ["annual_spending", "purchase_frequency", "age"]


def load_real_mall_data():
    raw = pd.read_csv(MALL_URL)

    def find_column(*keywords):
        for original in raw.columns:
            lowered = original.lower()
            if all(keyword in lowered for keyword in keywords):
                return original
        raise KeyError(f"No column matching {keywords} found in {list(raw.columns)}")

    rename_map = {
        find_column("income"): "annual_spending",
        find_column("spending"): "purchase_frequency",
        find_column("age"): "age",
        find_column("gen"): "gender",
    }
    return raw.rename(columns=rename_map)[["annual_spending", "purchase_frequency", "age", "gender"]]


def generate_fallback_segmentation_data(n=200, seed=RANDOM_STATE):
    """Synthetic stand-in matching the real Mall Customer dataset's
    published ranges (Age 18-70, Annual Income $15k-$137k, Spending Score
    1-99), built as three loosely-separated groups so the elbow method
    and K-Means still produce an interpretable result."""
    rng = np.random.default_rng(seed)
    n_per_group = n // 3
    groups = []
    profiles = [
        (30, 8, 25, 12, 40, 12),
        (55, 12, 55, 15, 32, 8),
        (95, 15, 80, 12, 28, 6),
    ]
    for income_mu, income_sd, spend_mu, spend_sd, age_mu, age_sd in profiles:
        groups.append(pd.DataFrame({
            "annual_spending": rng.normal(income_mu, income_sd, n_per_group).clip(15, 137).round(1),
            "purchase_frequency": rng.normal(spend_mu, spend_sd, n_per_group).clip(1, 99).round().astype(int),
            "age": rng.normal(age_mu, age_sd, n_per_group).clip(18, 70).round().astype(int),
            "gender": rng.choice(["Male", "Female"], n_per_group),
        }))
    df = pd.concat(groups, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_segmentation_data():
    try:
        df = load_real_mall_data()
        print(f"Loaded {len(df)} REAL customers from the Mall Customer Segmentation dataset ({MALL_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real Mall Customer dataset ({exc}). "
              f"Falling back to a documented synthetic dataset that matches its real value ranges.")
        df = generate_fallback_segmentation_data()
        print(f"Generated {len(df)} synthetic rows as a fallback.")
        return df


def run_part3_customer_segmentation():
    section("PART 3: CUSTOMER SEGMENTATION")
    df = load_segmentation_data()

    X = df[SEGMENTATION_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    K = range(1, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, "bo-")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.savefig("elbow_plot.png")
    plt.close()
    print("Saved elbow_plot.png -- look for the 'bend' (usually K=3 to 5 for this data).")

    # --- Justify the choice of K (required by the assignment) ---
    inertia_by_k = dict(zip(K, inertia))
    print("\nInertia by K (elbow method):")
    for k in K:
        print(f"  K={k}: inertia={inertia_by_k[k]:.1f}")

    k_list = list(K)
    print("\nPercent decrease in inertia at each step:")
    pct_drops = {}
    for prev_k, curr_k in zip(k_list[:-1], k_list[1:]):
        pct_drops[curr_k] = (inertia_by_k[prev_k] - inertia_by_k[curr_k]) / inertia_by_k[prev_k] * 100
        print(f"  K={prev_k} -> K={curr_k}: {pct_drops[curr_k]:.1f}% decrease")

    optimal_k = 3
    drop_after_optimal = pct_drops[optimal_k + 1]
    print(f"\nJustification for K={optimal_k}: the biggest single jump in inertia reduction happens from K=1 "
          f"to K=2 ({pct_drops[2]:.1f}%), and the curve keeps flattening after that -- by K={optimal_k + 1} the "
          f"per-step improvement has fallen to {drop_after_optimal:.1f}%, meaning each additional cluster "
          f"beyond {optimal_k} buys progressively less. We choose K={optimal_k} because it is the simplest "
          f"model where the resulting clusters are still clearly distinct and easy to act on (see the cluster "
          f"analysis and marketing strategies below).")

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = df.groupby("cluster")[SEGMENTATION_FEATURES].mean().round(2)
    cluster_counts = df["cluster"].value_counts().sort_index()
    print("\nCluster Characteristics (mean values):")
    print(cluster_summary)
    print("\nCluster sizes:")
    print(cluster_counts)

    print("\nTargeted Marketing Strategies:")
    for cluster in range(optimal_k):
        spending = cluster_summary.loc[cluster, "annual_spending"]
        frequency = cluster_summary.loc[cluster, "purchase_frequency"]
        print(f"\nCluster {cluster} (n={cluster_counts[cluster]}, "
              f"avg income=${spending:.0f}k, avg spending score={frequency:.0f}):")
        if spending > 70 and frequency > 60:
            print("  High-income, high-spending customers -> exclusive promotions, "
                  "VIP loyalty perks, early access to new products.")
        elif frequency > 60:
            print("  Frequent/high spending-score shoppers -> loyalty points, "
                  "bundle discounts to keep engagement high.")
        elif spending > 70:
            print("  High income but lower spending score -> targeted campaigns "
                  "highlighting premium products they aren't buying yet.")
        else:
            print("  Lower engagement customers -> re-engagement emails, "
                  "first-purchase discounts, awareness campaigns.")

    df.to_csv("customer_segments.csv", index=False)
    print("\nSaved cluster assignments to customer_segments.csv")


# ======================================================================
# EXTRA CREDIT: HOUSING DEMAND FORECASTING
#
# DATA SOURCE (real data): "Existing Home Sales" (not seasonally
# adjusted), series EXHOSLUSM495N, via FRED (Federal Reserve Bank of St.
# Louis): https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXHOSLUSM495N
# Series page: https://fred.stlouisfed.org/series/EXHOSLUSM495N
#
# ASSUMPTIONS: recent history (~10 years) is representative enough to
# extrapolate from; a trend + month-of-year seasonal model is a reasonable
# balance of simplicity and accuracy for a 6-month forecast; no exogenous
# shocks (rate changes, recessions) occur in the forecast window.
#
# CHALLENGES: plain linear regression on a raw time index (as in the
# original starter code) ignores seasonality and badly underfits a series
# that swings a lot by calendar month; housing demand also depends on
# mortgage rates, employment, and supply, none of which are in this
# single-series dataset.
#
# POTENTIAL IMPROVEMENTS: add exogenous features (mortgage rates,
# inventory, employment); use a model built for time series (SARIMA,
# Prophet, gradient boosting on lag features); cross-validate on rolling
# time windows instead of a single train/test split, since a normal
# train_test_split shuffle leaks future information into the past.
# ======================================================================

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXHOSLUSM495N"
FORECAST_HORIZON = 6


def load_real_fred_data():
    raw = pd.read_csv(FRED_URL)
    raw.columns = ["date", "sales"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.dropna().sort_values("date").reset_index(drop=True)
    cutoff = raw["date"].max() - pd.DateOffset(years=10)
    return raw[raw["date"] >= cutoff].reset_index(drop=True)


def generate_fallback_forecast_data(n_months=120, seed=RANDOM_STATE):
    """Synthetic stand-in resembling the real Existing Home Sales series:
    a mild trend plus a summer-peak/winter-trough seasonal pattern."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=n_months, freq="MS")
    t = np.arange(n_months)
    trend = 430 - 0.35 * t
    month_of_year = np.asarray(dates.month, dtype=float)
    seasonal = 70 * np.sin(2 * np.pi * (month_of_year - 3) / 12)
    noise = rng.normal(0, 15, n_months)
    sales = np.clip(trend + seasonal + noise, 200, None).round(1)
    return pd.DataFrame({"date": dates, "sales": sales})


def load_forecast_data():
    try:
        df = load_real_fred_data()
        print(f"Loaded {len(df)} REAL monthly observations from FRED's Existing Home Sales series ({FRED_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real FRED dataset ({exc}). "
              f"Falling back to a documented synthetic series that resembles its real trend and seasonality.")
        df = generate_fallback_forecast_data()
        print(f"Generated {len(df)} synthetic monthly rows as a fallback.")
        return df


def run_extra_credit_forecasting():
    section("EXTRA CREDIT: HOUSING DEMAND FORECASTING")
    df = load_forecast_data()
    df["month_index"] = np.arange(len(df))
    df["month_of_year"] = df["date"].dt.month.astype(str)

    X = df[["month_index", "month_of_year"]]
    y = df["sales"]

    baseline = LinearRegression()
    baseline.fit(df[["month_index"]], y)
    baseline_pred = baseline.predict(df[["month_index"]])
    print(f"\nBaseline (trend-only) training MAE: {mean_absolute_error(y, baseline_pred):.1f}")

    preprocessor = ColumnTransformer(
        transformers=[("season", OneHotEncoder(handle_unknown="ignore"), ["month_of_year"])],
        remainder="passthrough",
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(X, y)
    improved_pred = model.predict(X)
    print(f"Improved (trend + seasonality) training MAE: {mean_absolute_error(y, improved_pred):.1f}")

    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")
    future = pd.DataFrame({
        "month_index": np.arange(len(df), len(df) + FORECAST_HORIZON),
        "month_of_year": future_dates.month.astype(str),
    })
    future_predictions = model.predict(future)

    print(f"\nForecast for the next {FORECAST_HORIZON} months:")
    for d, p in zip(future_dates, future_predictions):
        print(f"  {d.strftime('%Y-%m')}: {p:,.1f} (thousand units)")

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], y, label="Historical Sales")
    plt.plot(df["date"], improved_pred, label="Fitted (trend + seasonality)", alpha=0.6)
    plt.plot(future_dates, future_predictions, label="Forecast (next 6 months)", linestyle="--", marker="o")
    plt.xlabel("Month")
    plt.ylabel("Existing Home Sales (thousand units)")
    plt.title("Housing Demand: Historical and 6-Month Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig("demand_forecast.png")
    plt.close()
    print("\nSaved plot to demand_forecast.png")

    # --- Assumptions, challenges, and potential improvements (required by
    # the assignment) -- printed here in addition to the module docstring
    # above, so they show up directly in the console output. ---
    print("\nAssumptions:")
    print("  - The last ~10 years of monthly data are representative enough of near-term demand to extrapolate from.")
    print("  - A model combining a long-run linear trend with a repeating month-of-year seasonal effect is a "
          "reasonable balance of simplicity and accuracy for a 6-month-ahead forecast.")
    print("  - No major exogenous shocks (interest-rate changes, policy changes, recessions) occur during the "
          "forecast window.")

    print("\nChallenges:")
    print(f"  - Plain linear regression on a raw time index alone (the original starter code's approach) "
          f"ignores seasonality: its training MAE was {mean_absolute_error(y, baseline_pred):.1f}, versus "
          f"{mean_absolute_error(y, improved_pred):.1f} once month-of-year seasonality is added.")
    print("  - Housing demand also depends on mortgage rates, employment, and housing supply, none of which "
          "are in this single-series dataset, so the model can only ever be a rough baseline.")
    print("  - Real-world series can have missing or revised months; the loader drops missing rows defensively.")

    print("\nPotential improvements:")
    print("  - Add exogenous features (mortgage rates, housing inventory, employment) as additional regression inputs.")
    print("  - Use a model built for time series (SARIMA, Prophet, or gradient boosting on lag features) "
          "instead of plain linear regression.")
    print("  - Cross-validate on rolling time windows instead of a single split, since a normal train/test "
          "split shuffle would leak future information into the past for time series data.")


if __name__ == "__main__":
    run_part1_house_prices()
    run_part2_customer_churn()
    run_part3_customer_segmentation()
    run_extra_credit_forecasting()
