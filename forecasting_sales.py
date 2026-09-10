"""
Extra Credit: AI-powered housing demand forecasting tool.
Author: Jacob Laplante

Reads historical housing sales/inventory data from a CSV file, trains a
regression model, and forecasts demand for the next 6 months.

DATA SOURCE (real data):
  "Existing Home Sales" (not seasonally adjusted), series EXHOSLUSM495N,
  published by the National Association of Realtors via the Federal
  Reserve Bank of St. Louis (FRED). Monthly count of existing homes sold
  in the U.S., in thousands of units -- a direct measure of housing demand.
  Official direct CSV download (no login required):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXHOSLUSM495N
  Series page / documentation:
    https://fred.stlouisfed.org/series/EXHOSLUSM495N

  NOTE ON THE SANDBOX THIS WAS DEVELOPED IN: outbound internet in the
  environment used to write/test this script is restricted to package
  registries only, so the live download above could not be exercised
  there. It will work normally in Google Colab, GitHub Actions, or your
  own laptop. If the download fails, the script falls back to a synthetic
  monthly series built to resemble the real one (an overall trend plus a
  realistic seasonal pattern -- home sales are typically highest in
  summer and lowest in winter) so it always runs end to end.

ASSUMPTIONS:
  - Recent history (last ~10 years of monthly data) is representative
    enough of near-term future demand to extrapolate from.
  - A model that captures (a) a long-run linear trend and (b) a repeating
    month-of-year seasonal effect is a reasonable balance of simplicity
    and accuracy for a 6-month-ahead forecast.
  - No exogenous shocks (rate changes, policy changes, recessions) occur
    in the forecast window -- a real limitation of any regression-only
    approach.

CHALLENGES:
  - Plain linear regression on a raw time index ('month = 1, 2, 3...'),
    as in the original starter code, completely ignores seasonality and
    badly underfits a series like home sales, which swings a lot by
    calendar month.
  - Housing demand is affected by mortgage rates, employment, and housing
    supply -- none of which are in this single-series dataset, so the
    model can only ever be a rough baseline.
  - Real-world data can include missing months or revisions; the loader
    below drops missing values defensively.

POTENTIAL IMPROVEMENTS:
  - Add exogenous features (mortgage rates, housing inventory, employment)
    as additional regression inputs.
  - Use a model built for time series (e.g. SARIMA, Prophet, or gradient
    boosting on lag features) instead of plain linear regression.
  - Cross-validate on rolling time windows instead of a single train/test
    split, since shuffling time series data (as a normal train_test_split
    would) leaks future information into the past.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXHOSLUSM495N"
FORECAST_HORIZON = 6
RANDOM_STATE = 42


def load_real_fred_data():
    """Download the real 'Existing Home Sales' series from FRED."""
    raw = pd.read_csv(FRED_URL)
    raw.columns = ["date", "sales"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.dropna().sort_values("date").reset_index(drop=True)
    # Keep roughly the most recent 10 years so the trend reflects current
    # market conditions rather than the full multi-decade history.
    cutoff = raw["date"].max() - pd.DateOffset(years=10)
    raw = raw[raw["date"] >= cutoff].reset_index(drop=True)
    return raw


def generate_fallback_data(n_months=120, seed=RANDOM_STATE):
    """Synthetic stand-in resembling the real Existing Home Sales series:
    a mild trend plus a summer-peak/winter-trough seasonal pattern, used
    only if the live download above is not reachable."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=n_months, freq="MS")

    t = np.arange(n_months)
    trend = 430 - 0.35 * t  # slow long-run decline, roughly matching recent real data
    month_of_year = np.asarray(dates.month, dtype=float)
    # Seasonal amplitude: home sales peak in summer (~+70k), trough in winter (~-70k)
    seasonal = 70 * np.sin(2 * np.pi * (month_of_year - 3) / 12)
    noise = rng.normal(0, 15, n_months)

    sales = np.clip(trend + seasonal + noise, 200, None).round(1)
    return pd.DataFrame({"date": dates, "sales": sales})


def load_data():
    try:
        df = load_real_fred_data()
        print(f"Loaded {len(df)} REAL monthly observations from FRED's "
              f"Existing Home Sales series ({FRED_URL}).")
        return df
    except Exception as exc:
        print(f"Could not download the real FRED dataset ({exc}). "
              f"Falling back to a documented synthetic series that "
              f"resembles its real trend and seasonality.")
        df = generate_fallback_data()
        print(f"Generated {len(df)} synthetic monthly rows as a fallback.")
        return df


df = load_data()
df["month_index"] = np.arange(len(df))          # linear time trend feature
df["month_of_year"] = df["date"].dt.month.astype(str)  # seasonal feature

X = df[["month_index", "month_of_year"]]
y = df["sales"]

# --- Baseline model: plain linear trend, like the original starter code ---
baseline = LinearRegression()
baseline.fit(df[["month_index"]], y)
baseline_pred = baseline.predict(df[["month_index"]])
print(f"\nBaseline (trend-only) training MAE: {mean_absolute_error(y, baseline_pred):.1f}")

# --- Improved model: trend + month-of-year seasonality ---
preprocessor = ColumnTransformer(
    transformers=[
        ("season", OneHotEncoder(handle_unknown="ignore"), ["month_of_year"]),
    ],
    remainder="passthrough",
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
model.fit(X, y)
improved_pred = model.predict(X)
print(f"Improved (trend + seasonality) training MAE: {mean_absolute_error(y, improved_pred):.1f}")

# Forecast the next 6 months
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

# Plot historical vs. forecasted results
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

# --- Assumptions, challenges, and potential improvements (required by the
# assignment) -- printed here at runtime in addition to the module
# docstring above, so they show up directly in the console output. ---
print("\nAssumptions:")
print("  - The last ~10 years of monthly data are representative enough of near-term demand to extrapolate from.")
print("  - A model combining a long-run linear trend with a repeating month-of-year seasonal effect is a "
      "reasonable balance of simplicity and accuracy for a 6-month-ahead forecast.")
print("  - No major exogenous shocks (interest-rate changes, policy changes, recessions) occur during the "
      "forecast window.")

print("\nChallenges:")
print(f"  - Plain linear regression on a raw time index alone (the original starter code's approach) ignores "
      f"seasonality: its training MAE was {mean_absolute_error(y, baseline_pred):.1f}, versus "
      f"{mean_absolute_error(y, improved_pred):.1f} once month-of-year seasonality is added -- seasonality "
      f"matters a lot for this series.")
print("  - Housing demand also depends on mortgage rates, employment, and housing supply, none of which are "
      "in this single-series dataset, so the model can only ever be a rough baseline.")
print("  - Real-world series can have missing or revised months; the loader drops missing rows defensively.")

print("\nPotential improvements:")
print("  - Add exogenous features (mortgage rates, housing inventory, employment) as additional regression inputs.")
print("  - Use a model built for time series (SARIMA, Prophet, or gradient boosting on lag features) instead "
      "of plain linear regression.")
print("  - Cross-validate on rolling time windows instead of a single split, since a normal train/test split "
      "shuffle would leak future information into the past for time series data.")
