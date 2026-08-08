"""
European Defense Readiness Monitor
-----------------------------------
An uncertainty-quantified index of defense readiness for NATO and
partner countries, built from live World Bank military-expenditure data.

Combines two components — current spending level and post-2022
trajectory — into a weighted composite score, then runs Monte Carlo
simulation over the weights to report confidence bands rather than
false precision.

Author: Kiril Mickovski
"""

import wbgapi as wb
import pandas as pd
import numpy as np


# ── Configuration ────────────────────────────────────────────────
COUNTRIES = ["POL", "LTU", "LVA", "EST", "USA", "NOR", "GRC", "GBR",
             "FRA", "DEU", "ITA", "ESP", "BEL", "NLD", "CAN"]

WEIGHT_SPENDING = 0.55   # weight on current spending level
WEIGHT_TRAJECTORY = 0.45  # weight on post-2022 ramp-up


# ── Data loading ─────────────────────────────────────────────────
def load_spending(start=2021, end=2025):
    """Pull military expenditure (% of GDP) from the World Bank."""
    df = wb.data.DataFrame("MS.MIL.XPND.GD.ZS", COUNTRIES, range(start, end))
    return df.dropna(how="all")


def build_components(df):
    """Derive the two scoring components from the spending panel.

    - spend: most recent year's military expenditure (% GDP)
    - traj:  change from 2021 to 2024 (post-invasion trajectory)
    """
    spend = df["YR2024"]
    traj = df["YR2024"] - df["YR2021"]
    return pd.DataFrame({"spend": spend, "traj": traj}).dropna()


# ── Scoring ──────────────────────────────────────────────────────
def normalize(s):
    """Min-max normalize a series to [0, 1], with divide-by-zero guard."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0 + 0.5
    return (s - lo) / (hi - lo)


def readiness_score(comp, w_spend=WEIGHT_SPENDING, w_traj=WEIGHT_TRAJECTORY):
    """Weighted composite readiness score, ranked high to low."""
    score = w_spend * normalize(comp["spend"]) + w_traj * normalize(comp["traj"])
    return score.sort_values(ascending=False).round(3)


# ── Uncertainty ──────────────────────────────────────────────────
def monte_carlo(comp, n_runs=2000, seed=42):
    """Run scoring n_runs times with random weights; return confidence bands.

    Draws weights from a Dirichlet distribution (they always sum to 1),
    recomputes the ranking each time, and reports how often each country
    lands in the top 3 / top 5, plus its average rank.
    """
    rng = np.random.default_rng(seed)
    n_spend, n_traj = normalize(comp["spend"]), normalize(comp["traj"])
    top3 = pd.Series(0, index=comp.index)
    top5 = pd.Series(0, index=comp.index)
    rank_sum = pd.Series(0.0, index=comp.index)

    for _ in range(n_runs):
        w = rng.dirichlet([5, 5])
        r = (w[0] * n_spend + w[1] * n_traj).rank(ascending=False)
        top3 += (r <= 3).astype(int)
        top5 += (r <= 5).astype(int)
        rank_sum += r

    return pd.DataFrame({
        "avg_rank": (rank_sum / n_runs).round(2),
        "top3_%": (100 * top3 / n_runs).round(1),
        "top5_%": (100 * top5 / n_runs).round(1),
    }).sort_values("avg_rank")


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("Fetching World Bank military expenditure data...\n")
    df = load_spending()
    comp = build_components(df)

    print("DEFENSE READINESS RANKING (weights 0.55 spend / 0.45 trajectory)")
    print("=" * 60)
    print(readiness_score(comp).to_string())

    print("\nMONTE CARLO CONFIDENCE (2000 runs, random weights)")
    print("=" * 60)
    print(monte_carlo(comp).to_string())


if __name__ == "__main__":
    main() 
    