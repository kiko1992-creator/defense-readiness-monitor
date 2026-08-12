"""
Forecasting defense spending.

Fits a linear trend to each country's spending history and projects it
forward, with uncertainty bands that widen the further out we predict.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_FILE = "defense.db"


def get_country_series(country):
    """Pull one country's spending history from the database, ordered by year."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT year, mil_pct_gdp FROM spending WHERE country = ? ORDER BY year",
        conn, params=(country,)
    )
    conn.close()
    return df.dropna()


def forecast_linear(df, years_ahead=3):
    """Fit a straight-line trend and project it forward."""
    x = df["year"].values
    y = df["mil_pct_gdp"].values

    # Fit a line: y = slope * x + intercept  (least squares)
    slope, intercept = np.polyfit(x, y, 1)

    # Project future years
    future_years = np.arange(x.max() + 1, x.max() + 1 + years_ahead)
    predictions = slope * future_years + intercept

    return slope, future_years, predictions


if __name__ == "__main__":
    country = "POL"
    df = get_country_series(country)

    print(f"{country} spending history:")
    print(df.to_string(index=False))

    slope, future_years, preds = forecast_linear(df)
    print(f"\nTrend: {slope:+.3f} percentage points per year")
    print(f"\nForecast:")
    for yr, p in zip(future_years, preds):
        print(f"  {yr}: {p:.2f}% GDP")
