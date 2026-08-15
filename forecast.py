"""
Forecasting defense spending with uncertainty quantification.

Uses per-country structural break detection to determine fitting windows,
rather than a single global cutoff year. Each country's most recent detected
break defines its post-break fitting window.

FORECASTING METHOD:
- Point forecast: Holt's damped trend (ExponentialSmoothing with damped_trend=True)
  This prevents unbounded extrapolation: steep trends gradually flatten rather
  than projecting to implausible values (e.g., 10% GDP or negative spending).
- Confidence intervals: Residual bootstrap using full historical volatility.

BREAK DETECTION:
Structural breaks are detected using a PELT-style algorithm (see detect_breaks.py).
The most recent break for each country determines where the "current regime" starts.

SPECIAL CASES:
- If no break detected in the last ~15 years (since 2010), falls back to
  full-history trend. Flag: used_regime_break=False
- Poland and Estonia: Algorithm detects 2023, but fitting from 2023 onward
  gives only 2 data points (2023, 2024), which is too thin for meaningful
  trend estimation. We start their window at 2022 instead. Justification:
  2022 already shows anticipatory movement in the data - Poland's value that
  year (2.20%) sits at the top of its 30-year range before the 2023 surge.
  This trades a small amount of regime purity for a minimally viable 3-point fit.
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DB_FILE = "defense.db"

# Countries from build_database.py
COUNTRIES = ["POL", "LTU", "LVA", "EST", "USA", "NOR", "GRC", "GBR",
             "FRA", "DEU", "ITA", "ESP", "BEL", "NLD", "CAN"]

# Detected breaks from detect_breaks.py (PELT algorithm, L2 cost, BIC penalty)
# These are the raw algorithm outputs - do not modify without documented reason
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

# Manual overrides with documented reasons
BREAK_OVERRIDES = {
    "POL": (2022, "2023 detected but only 2 points; 2022 shows anticipatory movement"),
    "EST": (2022, "2023 detected but only 2 points; 2022 shows anticipatory movement"),
}

RECENT_BREAK_THRESHOLD = 2010


def get_country_series(country):
    """Pull one country's full mil_pct_gdp history from the database."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        """
        SELECT year, value as mil_pct_gdp
        FROM indicators
        WHERE country = ? AND indicator = 'mil_pct_gdp'
        ORDER BY year
        """,
        conn, params=(country,)
    )
    conn.close()
    return df.dropna()


def get_break_year(country):
    """Determine the break year to use for this country."""
    raw_breaks = DETECTED_BREAKS.get(country, [])

    if country in BREAK_OVERRIDES:
        override_year, reason = BREAK_OVERRIDES[country]
        return override_year, True, reason

    if not raw_breaks:
        return None, False, "no breaks detected"

    most_recent = max(raw_breaks)

    if most_recent < RECENT_BREAK_THRESHOLD:
        return None, False, f"most recent break ({most_recent}) older than {RECENT_BREAK_THRESHOLD}"

    return most_recent, True, None


def fit_linear_trend(years, values):
    """Fit a linear trend, return slope, intercept, and residuals."""
    slope, intercept = np.polyfit(years, values, 1)
    fitted = slope * years + intercept
    residuals = values - fitted
    return slope, intercept, residuals


def forecast_holt_damped(values, n_forecast):
    """
    Fit Holt's damped trend model and forecast.

    Damped trend prevents unbounded extrapolation: the trend gradually
    flattens toward a horizontal asymptote rather than continuing linearly.
    This gives more realistic forecasts for both steep increases (Poland)
    and declining trends (Western Europe).
    """
    # Suppress convergence warnings for short series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Need at least 3 points for Holt's method
        if len(values) < 3:
            # Fall back to linear for very short series
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
            forecast = fit.forecast(n_forecast)
            return forecast
        except Exception:
            # Fall back to simple damped linear if Holt fails
            slope, intercept, _ = fit_linear_trend(np.arange(len(values)), values)
            last_fitted = slope * (len(values) - 1) + intercept
            # Apply damping factor of 0.85 per step
            phi = 0.85
            forecast = []
            cumulative_trend = 0
            for h in range(1, n_forecast + 1):
                cumulative_trend += slope * (phi ** h)
                forecast.append(last_fitted + cumulative_trend)
            return np.array(forecast)


def bootstrap_forecast_damped(years_full, values_full, years_trend, values_trend,
                               forecast_years, n_bootstrap=2000, ci=90):
    """
    Bootstrap confidence intervals around Holt's damped trend forecast.

    - Point forecast: Holt's damped trend on post-break data
    - CI: Resample residuals from full history, perturb trend data,
      refit damped model, collect forecast distribution
    """
    rng = np.random.default_rng(42)
    n_forecast = len(forecast_years)

    # Point forecast from damped trend
    point_forecast = forecast_holt_damped(values_trend, n_forecast)

    # Compute residuals from full history for realistic volatility
    _, _, residuals_full = fit_linear_trend(
        np.arange(len(values_full)), values_full
    )

    # Bootstrap
    n_trend = len(values_trend)
    bootstrap_forecasts = np.zeros((n_bootstrap, n_forecast))

    for b in range(n_bootstrap):
        # Resample residuals from full history
        resampled_residuals = rng.choice(residuals_full, size=n_trend, replace=True)

        # Perturb the trend data
        synthetic_y = values_trend + resampled_residuals

        # Refit damped model on perturbed data
        bootstrap_forecasts[b, :] = forecast_holt_damped(synthetic_y, n_forecast)

    # Compute confidence intervals
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = np.percentile(bootstrap_forecasts, lower_pct, axis=0)
    ci_upper = np.percentile(bootstrap_forecasts, upper_pct, axis=0)

    # Also compute linear forecast for comparison
    slope, intercept, _ = fit_linear_trend(
        np.arange(len(values_trend)), values_trend
    )
    linear_forecast = slope * (np.arange(n_forecast) + len(values_trend)) + intercept

    return point_forecast, ci_lower, ci_upper, linear_forecast


def forecast_country(country):
    """Generate forecast for one country using per-country break detection."""
    df = get_country_series(country)

    if len(df) < 2:
        return None

    years_full = df["year"].values
    values_full = df["mil_pct_gdp"].values

    break_year, used_regime_break, deviation_reason = get_break_year(country)

    raw_breaks = DETECTED_BREAKS.get(country, [])
    raw_most_recent = max(raw_breaks) if raw_breaks else None

    if used_regime_break and break_year is not None:
        mask = df["year"] >= break_year
        df_trend = df[mask]
        years_trend = df_trend["year"].values
        values_trend = df_trend["mil_pct_gdp"].values
        fitting_window = f"{break_year}-2024"
    else:
        years_trend = years_full
        values_trend = values_full
        fitting_window = f"{years_full.min()}-{years_full.max()}"

    n_points = len(years_trend)

    forecast_years = np.arange(2025, 2031)
    point, ci_low, ci_high, linear = bootstrap_forecast_damped(
        years_full, values_full, years_trend, values_trend,
        forecast_years, n_bootstrap=2000, ci=90
    )

    return {
        "country": country,
        "full_history": f"{years_full.min()}-{years_full.max()}",
        "n_full": len(years_full),
        "raw_detected_break": raw_most_recent,
        "break_used": break_year if used_regime_break else None,
        "used_regime_break": used_regime_break,
        "deviation_reason": deviation_reason,
        "fitting_window": fitting_window,
        "n_points": n_points,
        "latest_value": values_full[-1],
        "forecast_years": forecast_years,
        "point": point,
        "linear": linear,  # For before/after comparison
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main():
    print("=" * 95)
    print("DEFENSE SPENDING FORECASTS: Linear vs. Holt's Damped Trend")
    print("=" * 95)
    print("\nDamped trend prevents unbounded extrapolation:")
    print("- Steep upward trends (POL/EST) flatten toward plausible levels")
    print("- Declining trends (CAN/GBR/BEL) don't go negative")
    print("- Applied uniformly to all countries\n")

    results = []
    for country in COUNTRIES:
        result = forecast_country(country)
        if result:
            results.append(result)

    # Sort: regime-break countries first, then alphabetical
    results.sort(key=lambda x: (not x["used_regime_break"], x["country"]))

    # Before/After comparison for 2030
    print("-" * 95)
    print(f"{'Country':<6} {'Latest':>7} {'Regime?':<8} | "
          f"{'Linear 2030':>12} {'Damped 2030':>12} {'Change':>10} | {'Direction'}")
    print("-" * 95)

    for r in results:
        linear_2030 = r["linear"][5]  # Index 5 = 2030
        damped_2030 = r["point"][5]
        change = damped_2030 - linear_2030
        regime = "Yes" if r["used_regime_break"] else "No"

        # Determine what damping did
        if abs(change) < 0.1:
            direction = "~same"
        elif linear_2030 > r["latest_value"] and change < 0:
            direction = "pulled down (was too high)"
        elif linear_2030 < r["latest_value"] and change > 0:
            direction = "pulled up (was too low)"
        elif linear_2030 < 0 and damped_2030 > 0:
            direction = "prevented negative"
        elif change < 0:
            direction = "dampened rise"
        else:
            direction = "dampened fall"

        print(f"{r['country']:<6} {r['latest_value']:>6.2f}% {regime:<8} | "
              f"{linear_2030:>11.2f}% {damped_2030:>11.2f}% {change:>+9.2f}% | {direction}")

    print("-" * 95)

    # Full forecasts with damped trend
    print("\n" + "=" * 95)
    print("DAMPED TREND FORECASTS (2025-2030, 90% CI)")
    print("=" * 95)

    print(f"\n{'Country':<6} {'Latest':>7} | "
          f"{'2025':>18} {'2027':>18} {'2030':>18}")
    print("-" * 95)

    for r in results:
        def fmt_forecast(idx):
            return f"{r['point'][idx]:.2f} [{r['ci_low'][idx]:.1f}-{r['ci_high'][idx]:.1f}]"

        regime_marker = "*" if r["used_regime_break"] else " "
        print(f"{r['country']:<5}{regime_marker} {r['latest_value']:>6.2f}% | "
              f"{fmt_forecast(0):>18} {fmt_forecast(2):>18} {fmt_forecast(5):>18}")

    print("-" * 95)
    print("* = using post-break regime; unmarked = full history trend")

    # Highlight the key improvements
    print("\n" + "=" * 95)
    print("KEY IMPROVEMENTS FROM DAMPING")
    print("=" * 95)

    print("\nPreviously problematic forecasts (Linear -> Damped):")
    problematic = [
        ("POL", "was projecting ~10% (wartime level)"),
        ("EST", "was projecting ~7% (unrealistic)"),
        ("CAN", "was going negative"),
        ("GBR", "was approaching zero"),
        ("BEL", "was approaching zero"),
    ]

    for country, issue in problematic:
        r = next(x for x in results if x["country"] == country)
        print(f"  {country}: {r['linear'][5]:.2f}% -> {r['point'][5]:.2f}% ({issue})")


if __name__ == "__main__":
    main()
