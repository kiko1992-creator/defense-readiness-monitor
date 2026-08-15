"""
Export data for Power BI visualization.

Produces two CSV files in powerbi_export/:
1. indicators_fact.csv - Full long-format indicators with forecasts appended
2. countries_dim.csv - Country dimension table with cluster and region labels

CSV format: UTF-8 encoding, plain text, ready for Power BI Get Data > Text/CSV.
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DB_FILE = "defense.db"
OUTPUT_DIR = "powerbi_export"

# Countries and their regions (manually assigned based on geography/cluster analysis)
# Cluster 0: USA (North America, global superpower - outlier)
# Cluster 1: BEL, DEU, CAN, NLD, NOR, GBR (Western Europe + Canada)
# Cluster 2: ESP, FRA, GRC, ITA (Southern/Western Europe)
# Cluster 3: EST, LVA, LTU, POL (Baltic + Poland - frontline states)
COUNTRY_REGIONS = {
    "USA": "North America",
    "CAN": "North America",
    "GBR": "Western Europe",
    "DEU": "Western Europe",
    "FRA": "Western Europe",
    "NLD": "Western Europe",
    "BEL": "Western Europe",
    "NOR": "Northern Europe",
    "ESP": "Southern Europe",
    "ITA": "Southern Europe",
    "GRC": "Southern Europe",
    "POL": "Eastern Europe",
    "EST": "Baltic",
    "LVA": "Baltic",
    "LTU": "Baltic",
}

# Detected breaks from detect_breaks.py (for forecasting)
DETECTED_BREAKS = {
    "USA": [1951, 1955, 1971, 1993],
    "NOR": [1952, 1955, 1978, 1996],
    "GRC": [1958, 1974, 1989],
    "ITA": [1957, 1970, 1994],
    "ESP": [1979, 1991, 2002],
    "CAN": [1951, 1959, 1969],
    "POL": [1989, 2023],
    "LVA": [2002, 2018],
    "EST": [2001, 2023],
    "GBR": [1964, 1993],
    "FRA": [1965, 1995],
    "DEU": [1968, 1991],
    "BEL": [1955, 1992],
    "NLD": [1965, 1993],
    "LTU": [2017],
}

BREAK_OVERRIDES = {
    "POL": 2022,
    "EST": 2022,
}

RECENT_BREAK_THRESHOLD = 2010


def get_cluster_assignments():
    """Compute k-means cluster assignments for each country."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    # Filter to 2019-2024 and reliable indicators
    df = df[(df["year"] >= 2019) & (df["year"] <= 2024)]
    n_country_years = df[["country", "year"]].drop_duplicates().shape[0]
    coverage = df.dropna(subset=["value"]).groupby("indicator").size() / n_country_years
    keep = coverage[coverage >= 0.8].index
    df = df[df["indicator"].isin(keep)]

    profiles = df.groupby(["country", "indicator"])["value"].mean().unstack()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles)

    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    return dict(zip(profiles.index, labels))


def get_break_year(country):
    """Determine the break year to use for forecasting."""
    if country in BREAK_OVERRIDES:
        return BREAK_OVERRIDES[country]

    raw_breaks = DETECTED_BREAKS.get(country, [])
    if not raw_breaks:
        return None

    most_recent = max(raw_breaks)
    if most_recent < RECENT_BREAK_THRESHOLD:
        return None

    return most_recent


def forecast_holt_damped(values, n_forecast):
    """Fit Holt's damped trend model and forecast."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if len(values) < 3:
            slope = values[-1] - values[0] if len(values) > 1 else 0
            last = values[-1]
            return np.array([last + slope * (i + 1) * 0.8**i for i in range(n_forecast)])

        try:
            model = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            return fit.forecast(n_forecast)
        except Exception:
            slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
            last = values[-1]
            phi = 0.85
            forecast = []
            cumulative_trend = 0
            for h in range(1, n_forecast + 1):
                cumulative_trend += slope * (phi ** h)
                forecast.append(last + cumulative_trend)
            return np.array(forecast)


def generate_forecasts():
    """Generate mil_pct_gdp forecasts for all countries."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(
        "SELECT country, year, value FROM indicators WHERE indicator = 'mil_pct_gdp' ORDER BY country, year",
        conn
    )
    conn.close()

    forecast_years = list(range(2025, 2031))
    forecast_rows = []

    for country in df["country"].unique():
        country_df = df[df["country"] == country].dropna()
        if len(country_df) < 2:
            continue

        years = country_df["year"].values
        values = country_df["value"].values

        # Determine fitting window based on break detection
        break_year = get_break_year(country)
        if break_year:
            mask = years >= break_year
            if mask.sum() >= 2:
                values_trend = values[mask]
            else:
                values_trend = values
        else:
            values_trend = values

        # Generate forecast
        point_forecast = forecast_holt_damped(values_trend, len(forecast_years))

        for yr, val in zip(forecast_years, point_forecast):
            forecast_rows.append({
                "country": country,
                "year": yr,
                "indicator": "mil_pct_gdp",
                "value": val,
                "source": "forecast"
            })

    return pd.DataFrame(forecast_rows)


def export_indicators_fact():
    """Export indicators fact table with historical and forecast data."""
    print("Loading historical indicators...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT country, year, indicator, value FROM indicators", conn)
    conn.close()

    # Add source column for historical data
    df["source"] = "historical"

    print("Generating forecasts...")
    forecasts = generate_forecasts()

    print("Combining historical and forecast data...")
    combined = pd.concat([df, forecasts], ignore_index=True)

    # Sort for cleaner output
    combined = combined.sort_values(["country", "indicator", "year"]).reset_index(drop=True)

    return combined


def export_countries_dim():
    """Export countries dimension table with cluster and region."""
    print("Computing cluster assignments...")
    clusters = get_cluster_assignments()

    rows = []
    for country in sorted(COUNTRY_REGIONS.keys()):
        rows.append({
            "country": country,
            "country_name": get_country_name(country),
            "region": COUNTRY_REGIONS[country],
            "cluster": clusters.get(country, -1),
            "cluster_name": get_cluster_name(clusters.get(country, -1)),
            "is_frontline": country in ["POL", "EST", "LVA", "LTU"],
            "is_nato_founder": country in ["USA", "CAN", "GBR", "FRA", "BEL", "NLD", "NOR", "ITA"],
        })

    return pd.DataFrame(rows)


def get_country_name(iso_code):
    """Convert ISO code to full country name."""
    names = {
        "USA": "United States",
        "CAN": "Canada",
        "GBR": "United Kingdom",
        "DEU": "Germany",
        "FRA": "France",
        "NLD": "Netherlands",
        "BEL": "Belgium",
        "NOR": "Norway",
        "ESP": "Spain",
        "ITA": "Italy",
        "GRC": "Greece",
        "POL": "Poland",
        "EST": "Estonia",
        "LVA": "Latvia",
        "LTU": "Lithuania",
    }
    return names.get(iso_code, iso_code)


def get_cluster_name(cluster_id):
    """Convert cluster ID to descriptive name."""
    names = {
        0: "Global Superpower",
        1: "Core Western Allies",
        2: "Southern Europe",
        3: "Frontline States",
    }
    return names.get(cluster_id, "Unknown")


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("EXPORTING DATA FOR POWER BI")
    print("=" * 60)

    # Export indicators fact table
    print("\n--- indicators_fact.csv ---")
    indicators = export_indicators_fact()
    indicators_path = os.path.join(OUTPUT_DIR, "indicators_fact.csv")
    indicators.to_csv(indicators_path, index=False, encoding="utf-8")
    print(f"Exported {len(indicators)} rows to {indicators_path}")
    print(f"  Historical: {len(indicators[indicators['source'] == 'historical'])} rows")
    print(f"  Forecast: {len(indicators[indicators['source'] == 'forecast'])} rows")

    # Export countries dimension table
    print("\n--- countries_dim.csv ---")
    countries = export_countries_dim()
    countries_path = os.path.join(OUTPUT_DIR, "countries_dim.csv")
    countries.to_csv(countries_path, index=False, encoding="utf-8")
    print(f"Exported {len(countries)} rows to {countries_path}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles ready for Power BI in: {OUTPUT_DIR}/")
    print("  - indicators_fact.csv (fact table)")
    print("  - countries_dim.csv (dimension table)")
    print("\nTo import: Get Data > Text/CSV > select file")

    # Preview
    print("\n--- Preview: countries_dim.csv ---")
    print(countries.to_string(index=False))


if __name__ == "__main__":
    main()
