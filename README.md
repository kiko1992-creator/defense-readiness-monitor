# Defense Readiness Monitor

A data-driven analysis of whether each NATO country's defense spending trajectory is consistent with its actual security exposure and alliance commitments.

Most defense comparisons rank countries by a single metric: how much they spend. This project asks a different question: *given a country's geographic position, governance quality, economic capacity, and threat environment, is it spending appropriately?* The Baltics and Poland face direct Russian exposure; Western European states do not. Do their spending profiles reflect that asymmetry?

## Findings

Two substantive results have emerged so far:

### 1. Rule of law negatively correlates with defense spending (r ≈ −0.43)

Countries with stronger rule-of-law scores (Germany, Netherlands, Norway) tend to spend *less* on defense as a percentage of GDP, while countries with weaker scores (Greece, Poland) spend more. This is a moderate, not strong, relationship, and it isn't uniform: the Baltic states have comparatively strong rule-of-law scores (Estonia 1.50, Lithuania 1.22) despite high spending, suggesting Greece and Poland specifically may be driving much of this pattern rather than a clean trend across all 15 countries.

### 2. The Baltics and Poland cluster together

K-means clustering (k=4) on the 10 reliable indicators groups **Estonia, Latvia, Lithuania, and Poland** into the same cluster. These four share a distinctive profile: high defense spending relative to GDP, lower GDP per capita, and direct exposure to Russia. The algorithm finds them more similar to each other than to any Western European or North American ally.

## Schema

The database uses a **long (tidy) format** rather than one column per indicator:

```
indicators
├── country     TEXT      (ISO 3-letter code)
├── year        INTEGER   (2019–2024)
├── indicator   TEXT      (e.g., "mil_pct_gdp", "rule_of_law")
└── value       REAL
PRIMARY KEY (country, year, indicator)
```

This design makes adding new data sources trivial: write a fetch function, append rows. The table shape never changes.

## Data sources

Data comes from the **World Bank API** (`wbgapi`) and **SIPRI Military Expenditure Database**:

| Indicator | Code | Source |
|-----------|------|--------|
| Military expenditure (% GDP), 1949–2018 | Share of GDP sheet | SIPRI |
| Military expenditure (% GDP), 2019–2024 | `MS.MIL.XPND.GD.ZS` | WDI |
| GDP growth (annual %) | `NY.GDP.MKTP.KD.ZG` | WDI |
| GDP per capita (current US$) | `NY.GDP.PCAP.CD` | WDI |
| Inflation (consumer prices, %) | `FP.CPI.TOTL.ZG` | WDI |
| Unemployment (% labor force) | `SL.UEM.TOTL.ZS` | WDI |
| Trade openness (% GDP) | `NE.TRD.GNFS.ZS` | WDI |
| Population | `SP.POP.TOTL` | WDI |
| Population growth (%) | `SP.POP.GROW` | WDI |
| Political stability | `GOV_WGI_PV.EST` | WGI |
| Rule of law | `GOV_WGI_RL.EST` | WGI |

**SIPRI** = Stockholm International Peace Research Institute ([sipri.org/databases/milex](https://sipri.org/databases/milex))
**WDI** = World Bank, World Development Indicators (database 2)
**WGI** = World Bank, Worldwide Governance Indicators (database 3)

## Countries

15 NATO members, chosen to span frontline, core, and peripheral alliance positions:

```
POL  LTU  LVA  EST  USA  NOR  GRC  GBR  FRA  DEU  ITA  ESP  BEL  NLD  CAN
```

## Excluded indicators

Three indicators are fetched but excluded from analysis due to insufficient coverage (<80% of country-years):

| Indicator | Problem | Detail |
|-----------|---------|--------|
| `battle_deaths` | Almost entirely missing | World Bank's `VC.BTL.DETH` series has real reported values for only 2 of 90 country-years for this set of countries—not enough to determine whether this reflects genuine absence of conflict or simply a reporting gap in the source. Would need a different data source (e.g., UCDP directly) to actually test conflict exposure. |
| `personnel` | Discontinued | World Bank only has data for 2019–2020; the series stopped updating. |
| `gov_debt_pct_gdp` | Sparse | Only 4 of 15 countries report through this indicator; most EU members use different reporting channels. |

These are retained in the database for completeness but filtered out before correlation and clustering analysis to avoid distorting results with near-empty columns.

## Running it

```bash
git clone https://github.com/kiko1992-creator/defense-readiness-monitor.git
cd defense-readiness-monitor

py -m venv venv
.\venv\Scripts\Activate        # Windows
pip install -r requirements.txt

# Build the database (fetches from World Bank API)
py build_database.py

# Run correlation analysis
py analyze_correlation.py

# Run clustering
py cluster_countries.py
```

## Outputs

- `defense.db` — SQLite database with all indicator data
- `correlation_heatmap.png` — Pearson correlation matrix across all indicators
- `cluster_elbow.png` — Elbow plot for choosing optimal k

## Limitations

- **Resourcing, not readiness.** Spending levels do not capture equipment quality, force training, logistics, or classified capability.
- **World Bank definitions.** Spending figures differ slightly from NATO and SIPRI figures for some countries.
- **Correlation is not causation.** The negative rule-of-law correlation may reflect omitted variables (geography, threat perception) rather than a direct relationship.

## Author

Kiril Mickovski
