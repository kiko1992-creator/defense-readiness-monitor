"""
Predict defense spending from economic, demographic, and governance indicators.

Trains a linear regression to predict mil_pct_gdp from the 9 other reliable
indicators (excluding battle_deaths, personnel, gov_debt_pct_gdp per the
coverage rule). Uses country-averaged profiles and leave-one-out cross-
validation to evaluate predictive accuracy.

This answers the question: how much of a country's defense spending can be
explained by its economic capacity, demographics, and governance quality?
Countries that deviate strongly from the model's prediction may be over- or
under-spending relative to their "peers" on these dimensions.
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

DB_FILE = "defense.db"
MIN_COVERAGE = 0.8

# Target variable
TARGET = "mil_pct_gdp"

# Excluded indicators (per README: insufficient coverage)
EXCLUDED = ["battle_deaths", "personnel", "gov_debt_pct_gdp"]


def load_country_profiles():
    """Build one row per country: each indicator averaged across 2019-2024.

    We use only 2019-2024 because that's the period where all indicators
    have coverage. mil_pct_gdp has 1949-2024 data from SIPRI backfill, but
    the other indicators (GDP, governance, etc.) only have 2019-2024.
    """
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    # Filter to 2019-2024 period where all indicators have data
    df = df[(df["year"] >= 2019) & (df["year"] <= 2024)]

    # Calculate coverage within this period (15 countries × 6 years = 90 max)
    n_country_years = df[["country", "year"]].drop_duplicates().shape[0]
    coverage = df.dropna(subset=["value"]).groupby("indicator").size() / n_country_years
    keep = coverage[coverage >= MIN_COVERAGE].index
    df = df[df["indicator"].isin(keep)]

    # Also exclude the known problematic ones explicitly
    df = df[~df["indicator"].isin(EXCLUDED)]

    # Average each indicator across the 6 years, per country
    profiles = df.groupby(["country", "indicator"])["value"].mean().unstack()
    return profiles


def run_loocv(X, y, country_names):
    """
    Leave-one-out cross-validation: train on 14 countries, predict the 15th.

    Uses Ridge regression with regularization because we have only 15 samples
    and 9 features - OLS would severely overfit. Ridge penalizes large
    coefficients, preventing the wild extrapolations that occur when the
    held-out country has extreme values on some feature.
    """
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))

    # Ridge regularization strength candidates
    alphas = [0.1, 1.0, 10.0, 100.0]

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use RidgeCV to find best alpha within each fold
        model = RidgeCV(alphas=alphas)
        model.fit(X_train_scaled, y_train)
        predictions[test_idx] = model.predict(X_test_scaled)

    errors = predictions - y
    return predictions, errors


def interpret_coefficient(feature, coef, std):
    """Generate plain-language interpretation of a coefficient."""
    direction = "increase" if coef > 0 else "decrease"
    abs_coef = abs(coef)

    # Adjust interpretation based on feature type
    if feature == "population":
        # Population is in absolute numbers, so standardized coef is more meaningful
        return (f"A 1-std increase in population is associated with a "
                f"{abs_coef:.3f} pp {direction} in defense spending")
    elif feature in ["gdp_per_capita"]:
        return (f"A 1-std increase in GDP per capita is associated with a "
                f"{abs_coef:.3f} pp {direction} in defense spending")
    elif feature in ["rule_of_law", "political_stability"]:
        return (f"A 1-std increase in {feature.replace('_', ' ')} is associated with a "
                f"{abs_coef:.3f} pp {direction} in defense spending")
    elif "_pct_" in feature or feature.endswith("_growth"):
        return (f"A 1-std increase in {feature.replace('_', ' ')} is associated with a "
                f"{abs_coef:.3f} pp {direction} in defense spending")
    else:
        return (f"A 1-std increase in {feature.replace('_', ' ')} is associated with a "
                f"{abs_coef:.3f} pp {direction} in defense spending")


def analyze_worst_prediction(country, actual, predicted, profiles):
    """Explain why a country might be poorly predicted."""
    error = predicted - actual
    direction = "overpredicts" if error > 0 else "underpredicts"

    print(f"\n  The model {direction} {country}'s spending by {abs(error):.2f} pp")
    print(f"  Actual: {actual:.2f}%, Predicted: {predicted:.2f}%")

    # Get this country's profile
    country_profile = profiles.loc[country]

    # Compare to mean
    means = profiles.mean()

    print(f"\n  {country}'s notable deviations from the 15-country average:")
    deviations = (country_profile - means) / profiles.std()
    for feature in deviations.abs().sort_values(ascending=False).head(3).index:
        if feature == TARGET:
            continue
        dev = deviations[feature]
        direction = "above" if dev > 0 else "below"
        print(f"    - {feature}: {abs(dev):.1f} std {direction} average")


def main():
    print("=" * 75)
    print("PREDICTING DEFENSE SPENDING FROM COUNTRY PROFILES")
    print("=" * 75)

    # Load data
    profiles = load_country_profiles()
    print(f"\nLoaded {profiles.shape[0]} countries, {profiles.shape[1]} indicators")
    print(f"Target: {TARGET}")
    print(f"Features: {[c for c in profiles.columns if c != TARGET]}")

    # Separate features and target
    feature_cols = [c for c in profiles.columns if c != TARGET]
    X = profiles[feature_cols].values
    y = profiles[TARGET].values
    country_names = profiles.index.values

    # Run LOOCV
    print("\n" + "-" * 75)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("-" * 75)

    predictions, errors = run_loocv(X, y, country_names)

    # Full-sample metrics
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = median_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    # USA-excluded metrics (to show how much one outlier drives the result)
    usa_idx = np.where(country_names == "USA")[0][0]
    mask_no_usa = np.ones(len(y), dtype=bool)
    mask_no_usa[usa_idx] = False

    rmse_no_usa = np.sqrt(mean_squared_error(y[mask_no_usa], predictions[mask_no_usa]))
    mae_no_usa = median_absolute_error(y[mask_no_usa], predictions[mask_no_usa])
    r2_no_usa = r2_score(y[mask_no_usa], predictions[mask_no_usa])

    print(f"\n{'Metric':<25} {'All 15':>12} {'Excluding USA':>15}")
    print("-" * 55)
    print(f"{'RMSE (pp)':<25} {rmse:>12.3f} {rmse_no_usa:>15.3f}")
    print(f"{'Median Absolute Error':<25} {mae:>12.3f} {mae_no_usa:>15.3f}")
    print(f"{'R-squared':<25} {r2:>12.3f} {r2_no_usa:>15.3f}")

    print(f"\nInterpretation:")
    print(f"  - Full sample R-squared is {r2:.2f} (negative = worse than predicting the mean)")
    print(f"  - Excluding USA alone improves R-squared to {r2_no_usa:.2f}")
    print(f"  - USA is such an extreme outlier that it dominates the error metric")
    print(f"  - Even without USA, economic/governance indicators explain little variance")

    # Fit final model on all data for coefficient interpretation
    print("\n" + "-" * 75)
    print("MODEL COEFFICIENTS (Ridge regression, standardized features)")
    print("-" * 75)
    print("\nNote: With only 15 countries and 9 features, we use Ridge regression")
    print("to prevent overfitting. Coefficients are shrunk toward zero.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use RidgeCV to find optimal regularization
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model.fit(X_scaled, y)
    print(f"\nOptimal regularization strength (alpha): {model.alpha_}")

    # Sort coefficients by absolute magnitude
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
        "abs_coef": np.abs(model.coef_)
    }).sort_values("abs_coef", ascending=False)

    print(f"\nIntercept: {model.intercept_:.2f}% (average spending when all features are at mean)")
    print("\nFeature coefficients (sorted by importance):\n")

    feature_stds = profiles[feature_cols].std()

    for _, row in coef_df.iterrows():
        feature = row["feature"]
        coef = row["coefficient"]
        std = feature_stds[feature]

        print(f"  {feature:<20} {coef:>+7.3f}")
        print(f"    -> {interpret_coefficient(feature, coef, std)}")
        print()

    # Per-country predictions
    print("-" * 75)
    print("PER-COUNTRY PREDICTIONS (LOOCV)")
    print("-" * 75)

    results = pd.DataFrame({
        "country": country_names,
        "actual": y,
        "predicted": predictions,
        "error": errors,
        "abs_error": np.abs(errors)
    }).sort_values("abs_error", ascending=False)

    print(f"\n{'Country':<8} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 40)
    for _, row in results.iterrows():
        print(f"{row['country']:<8} {row['actual']:>7.2f}% {row['predicted']:>9.2f}% {row['error']:>+7.2f}")

    # Analyze worst prediction
    print("\n" + "-" * 75)
    print("WORST PREDICTION ANALYSIS")
    print("-" * 75)

    worst = results.iloc[0]
    print(f"\nWorst predicted country: {worst['country']}")
    analyze_worst_prediction(
        worst["country"],
        worst["actual"],
        worst["predicted"],
        profiles
    )

    # Additional context based on what we know
    worst_country = worst["country"]
    print(f"\n  Interpretation:")

    if worst_country == "GRC":
        print("    Greece has historically high defense spending driven by tensions")
        print("    with Turkey (a fellow NATO member), which the model's indicators")
        print("    don't capture. Its spending reflects a regional security dynamic")
        print("    that economic and governance variables can't explain.")
    elif worst_country == "POL":
        print("    Poland's spending reflects its position as a frontline state")
        print("    facing Russia directly. The model's economic/governance indicators")
        print("    don't capture geographic threat exposure.")
    elif worst_country == "USA":
        print("    The US is a global superpower with defense commitments far beyond")
        print("    what its domestic economic profile would suggest. The model can't")
        print("    capture alliance leadership responsibilities.")
    elif worst_country in ["EST", "LVA", "LTU"]:
        print(f"    {worst_country} is a Baltic state with direct Russian exposure.")
        print("    Small population and economy, but existential threat perception")
        print("    drives spending higher than economic indicators predict.")
    else:
        print(f"    {worst_country}'s deviation may reflect country-specific security")
        print("    considerations not captured by economic/governance indicators.")


if __name__ == "__main__":
    main()
