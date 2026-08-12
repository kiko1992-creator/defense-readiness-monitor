"""
Correlation analysis: which indicators actually move together with
military spending (mil_pct_gdp)?

Reads the long-format 'indicators' table, pivots it to wide (one row per
country-year, one column per indicator), computes correlations, and saves
a heatmap.
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves files, never opens a window
import matplotlib.pyplot as plt
import seaborn as sns

DB_FILE = "defense.db"


def load_wide(min_coverage=0.8):
    """Read the long table, drop indicators with too much missing data, and
    reshape to wide: one column per indicator.

    min_coverage: an indicator must have real (non-null) values for at
    least this fraction of country-years to be included. Sparse indicators
    (like personnel, only 2019-2020, or gov_debt_pct_gdp, only 4 countries)
    would distort correlations if included, since pandas' pairwise-complete
    correlation would silently compute them from a tiny, unrepresentative
    subset of rows rather than the full 90.
    """
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    n_country_years = df[["country", "year"]].drop_duplicates().shape[0]
    coverage = df.dropna(subset=["value"]).groupby("indicator").size() / n_country_years

    keep = coverage[coverage >= min_coverage].index
    dropped = coverage[coverage < min_coverage]
    if len(dropped) > 0:
        print("Excluded from correlation analysis (insufficient coverage):")
        for name, pct in dropped.items():
            print(f"  - {name}: {pct:.0%} coverage")

    df = df[df["indicator"].isin(keep)]
    df_wide = df.pivot(index=["country", "year"], columns="indicator", values="value")
    return df_wide


def main():
    df_wide = load_wide()
    print(f"Wide table shape: {df_wide.shape[0]} rows, {df_wide.shape[1]} columns")

    # Pearson: measures LINEAR relationships (assumes roughly straight-line trend)
    pearson = df_wide.corr(method="pearson")

    # Spearman: measures MONOTONIC relationships (one goes up, other goes up --
    # but not necessarily in a straight line). More robust to outliers, and you
    # already used this in your Colab work.
    spearman = df_wide.corr(method="spearman")

    # Focus on what actually correlates with mil_pct_gdp specifically, sorted
    # strongest to weakest. This is the direct answer to "what moves with
    # defense spending."
    print("\nPearson correlation with mil_pct_gdp, strongest first:")
    print(pearson["mil_pct_gdp"].drop("mil_pct_gdp").sort_values(key=abs, ascending=False))

    print("\nSpearman correlation with mil_pct_gdp, strongest first:")
    print(spearman["mil_pct_gdp"].drop("mil_pct_gdp").sort_values(key=abs, ascending=False))

    # Heatmap of the full Pearson correlation matrix, saved as an image file
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True)
    plt.title("Correlation matrix: all indicators")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("\nSaved correlation_heatmap.png")


if __name__ == "__main__":
    main()