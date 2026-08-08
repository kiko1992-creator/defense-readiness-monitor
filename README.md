# European Defense Readiness Monitor

An uncertainty-quantified index of defense readiness for NATO and partner
countries, built from live World Bank military-expenditure data.

Most defense-spending comparisons rank countries by a single number — how
much they spend. This tool does two things differently: it combines *spending
level* with *spending trajectory* (who actually ramped up after 2022), and it
runs a Monte Carlo simulation over the scoring weights to report **confidence
bands** instead of false precision. The result distinguishes rankings that are
robust from rankings that are essentially coin-flips.

## Key findings

- **Poland and Estonia lead in 100% of weight scenarios** — their top-tier
  position is robust to any reasonable weighting.
- **The United States ranks 5th, not top-3** — high spending, but a flat
  post-2022 trajectory places it behind the frontline states that actually
  surged after Russia's invasion.
- **Mid-table rankings (6th–15th) are not statistically distinguishable** —
  the model reports these as uncertain rather than inventing a false order.

## How it works

The readiness score combines two normalized components:

| Component | Weight | Source |
|-----------|--------|--------|
| Current spending (% GDP, 2024) | 0.55 | World Bank |
| Trajectory (change 2021 → 2024) | 0.45 | World Bank |

Both are normalized to [0, 1], then combined. A Monte Carlo simulation
(2,000 runs) redraws the weights from a Dirichlet distribution and re-ranks
each time, reporting how often each country lands in the top 3 and top 5.

## Running it

```bash
# Clone the repository
git clone https://github.com/kiko1992-creator/defense-readiness-monitor.git
cd defense-readiness-monitor

# Set up the environment
py -m venv venv
.\venv\Scripts\Activate        # Windows
pip install -r requirements.txt

# Run
py readiness_model.py
```

The script fetches live data from the World Bank API, so an internet
connection is required.

## Data sources

- **World Bank** — Military expenditure (% of GDP), indicator `MS.MIL.XPND.GD.ZS`

## Limitations

This model measures *resourcing and responsiveness*, not operational
readiness. It does not capture equipment quality, force training, logistics,
or classified capability. Spending figures use the World Bank definition,
which differs slightly from NATO and SIPRI figures for some countries.

## Author

Kiril Mickovski
