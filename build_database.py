"""
Build the defense database.

Fetches indicators from the World Bank and stores them in a local SQLite
database (defense.db) using a LONG/TIDY schema: one row per
(country, year, indicator) fact, rather than one column per indicator.

Adding a new data source means adding a new fetch_* function and stacking
its output onto the rest -- the table shape itself never has to change.
"""

import sqlite3
import pandas as pd
import wbgapi as wb


COUNTRIES = ["POL", "LTU", "LVA", "EST", "USA", "NOR", "GRC", "GBR",
             "FRA", "DEU", "ITA", "ESP", "BEL", "NLD", "CAN"]

DB_FILE = "defense.db"


def fetch_indicator(indicator_code, indicator_name, start=2019, end=2025, db=2):
    """Pull one World Bank indicator as a tidy (country, year, indicator, value) table.

    db: which World Bank database the indicator lives in. Most indicators
    (spending, GDP, population, etc.) are in database 2, World Development
    Indicators -- the default. Governance indicators like PV.EST and RL.EST
    live in database 3, Worldwide Governance Indicators, and must be
    fetched with db=3 or the API returns a malformed response.
    """
    wb.db = db
    df = wb.data.DataFrame(indicator_code, COUNTRIES, range(start, end))
    df = df.stack().reset_index()
    df.columns = ["country", "year", "value"]
    df["year"] = df["year"].str.replace("YR", "").astype(int)
    df["indicator"] = indicator_name
    return df[["country", "year", "indicator", "value"]]


# --- Military ---------------------------------------------------------

def fetch_spending(start=2019, end=2025):
    """Military expenditure as % of GDP."""
    return fetch_indicator("MS.MIL.XPND.GD.ZS", "mil_pct_gdp", start, end)


def fetch_personnel(start=2019, end=2025):
    """Armed forces personnel, total headcount."""
    return fetch_indicator("MS.MIL.TOTL.P1", "personnel", start, end)


# --- Economic capacity --------------------------------------------------

def fetch_gdp_growth(start=2019, end=2025):
    """GDP growth, annual %."""
    return fetch_indicator("NY.GDP.MKTP.KD.ZG", "gdp_growth", start, end)


def fetch_gdp_per_capita(start=2019, end=2025):
    """GDP per capita, current US$."""
    return fetch_indicator("NY.GDP.PCAP.CD", "gdp_per_capita", start, end)


def fetch_debt(start=2019, end=2025):
    """Central government debt, total, % of GDP."""
    return fetch_indicator("GC.DOD.TOTL.GD.ZS", "gov_debt_pct_gdp", start, end)


def fetch_inflation(start=2019, end=2025):
    """Inflation, consumer prices, annual %."""
    return fetch_indicator("FP.CPI.TOTL.ZG", "inflation", start, end)


def fetch_unemployment(start=2019, end=2025):
    """Unemployment, total, % of labor force."""
    return fetch_indicator("SL.UEM.TOTL.ZS", "unemployment", start, end)


def fetch_trade_openness(start=2019, end=2025):
    """Trade (exports + imports) as % of GDP."""
    return fetch_indicator("NE.TRD.GNFS.ZS", "trade_pct_gdp", start, end)


# --- Demographic ----------------------------------------------------------

def fetch_population(start=2019, end=2025):
    """Total population."""
    return fetch_indicator("SP.POP.TOTL", "population", start, end)


def fetch_population_growth(start=2019, end=2025):
    """Population growth, annual %."""
    return fetch_indicator("SP.POP.GROW", "population_growth", start, end)


# --- Security / institutional ---------------------------------------------

def fetch_battle_deaths(start=2019, end=2025):
    """Battle-related deaths (sourced from UCDP/PRIO, distributed via World Bank)."""
    return fetch_indicator("VC.BTL.DETH", "battle_deaths", start, end)


def fetch_political_stability(start=2019, end=2025):
    """Political Stability and Absence of Violence/Terrorism (Worldwide Governance Indicators)."""
    return fetch_indicator("GOV_WGI_PV.EST", "political_stability", start, end, db=3)


def fetch_rule_of_law(start=2019, end=2025):
    """Rule of Law estimate (Worldwide Governance Indicators)."""
    return fetch_indicator("GOV_WGI_RL.EST", "rule_of_law", start, end, db=3)


# --- Assembly ---------------------------------------------------------

# Every fetch function to include when building the database. Adding a
# new source later means writing one more fetch_* function above and
# adding it to this list -- nothing else in build_database() changes.
FETCHERS = [
    fetch_spending,
    fetch_personnel,
    fetch_gdp_growth,
    fetch_gdp_per_capita,
    fetch_debt,
    fetch_inflation,
    fetch_unemployment,
    fetch_trade_openness,
    fetch_population,
    fetch_population_growth,
    fetch_battle_deaths,
    fetch_political_stability,
    fetch_rule_of_law,
]


def create_schema(conn):
    """Create the indicators table with an explicit composite primary key."""
    conn.execute("DROP TABLE IF EXISTS indicators")
    conn.execute("""
        CREATE TABLE indicators (
            country TEXT,
            year INTEGER,
            indicator TEXT,
            value REAL,
            PRIMARY KEY (country, year, indicator)
        )
    """)


def build_database():
    """Fetch every indicator, stack them, and write the result into SQLite."""
    print(f"Fetching {len(FETCHERS)} indicators from World Bank...")
    frames = []
    for fetch_fn in FETCHERS:
        name = fetch_fn.__name__
        print(f"  - {name}")
        frames.append(fetch_fn())

    df = pd.concat(frames, ignore_index=True)

    print(f"Writing {len(df)} rows to {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    create_schema(conn)
    df.to_sql("indicators", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("Done. Database built.")

    # Sanity check: how many ACTUAL non-null values does each indicator have?
    # (Row existence alone isn't enough -- a row can exist with value = NaN,
    # which is exactly what happened with battle_deaths.)
    n_country_years = df[["country", "year"]].drop_duplicates().shape[0]
    print(f"\nNon-null values per indicator (out of {n_country_years} possible country-years):")
    non_null_counts = df.dropna(subset=["value"]).groupby("indicator").size()
    all_indicators = df["indicator"].unique()
    non_null_counts = non_null_counts.reindex(all_indicators, fill_value=0)
    print(non_null_counts.sort_values(ascending=False))


if __name__ == "__main__":
    build_database()