"""
K-means clustering: do the 15 countries fall into natural groups based on
their overall profile across the 10 reliable indicators?

Unlike analyze_correlation.py (which worked at the country-year level),
clustering needs one row PER COUNTRY, so each year's indicators are
averaged first. Countries are standardized before clustering so no single
variable (like population, which is on a much larger scale) dominates.
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DB_FILE = "defense.db"
MIN_COVERAGE = 0.8  # same threshold used in analyze_correlation.py


def load_country_profiles():
    """Build one row per country: each indicator averaged across all years."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    n_country_years = df[["country", "year"]].drop_duplicates().shape[0]
    coverage = df.dropna(subset=["value"]).groupby("indicator").size() / n_country_years
    keep = coverage[coverage >= MIN_COVERAGE].index
    df = df[df["indicator"].isin(keep)]

    # Average each indicator across the 6 years, per country. This collapses
    # 90 country-year rows down to 15 country rows -- one profile per country.
    profiles = df.groupby(["country", "indicator"])["value"].mean().unstack()
    return profiles


def find_elbow(X_scaled, max_k=8):
    """Run k-means for k = 1..max_k and record inertia (within-cluster
    spread), so we can pick the k where adding more clusters stops helping
    much -- the 'elbow' of the curve."""
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow method: choosing k")
    plt.tight_layout()
    plt.savefig("cluster_elbow.png", dpi=150)
    print("Saved cluster_elbow.png -- look for where the curve bends (the 'elbow')")
    return inertias


def main():
    profiles = load_country_profiles()
    print(f"Country profiles: {profiles.shape[0]} countries, {profiles.shape[1]} indicators")
    print(profiles.round(2))

    # Standardize: mean 0, std 1 for every column, so no indicator dominates
    # the distance calculation just because it's on a bigger numeric scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles)

    find_elbow(X_scaled)

    # Fit the final model. k=3 is a starting guess -- adjust once you've
    # looked at cluster_elbow.png and picked the actual elbow point.
    k = 4
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    result = pd.DataFrame({"country": profiles.index, "cluster": labels})
    result = result.sort_values("cluster")
    print(f"\nCluster assignments (k={k}):")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()