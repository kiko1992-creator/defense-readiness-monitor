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


def fetch_indicator(indicator_code, indicator_name, start=2019, end=2025):
    """Pull one World Bank indicator as a tidy (country, year, indicator, value) table."""
    df = wb.data.DataFrame(indicator_code, COUNTRIES, range(start, end))
    # Reshape from wide (year columns) to long (one row per country-year)
    df = df.stack().reset_index()
    df.columns = ["country", "year", "value"]
    df["year"] = df["year"].str.replace("YR", "").astype(int)
    # Tag every row with which indicator it belongs to -- this is the
    # column that lets us stack many indicators into one table safely.
    df["indicator"] = indicator_name
    # Reorder columns to match our schema: country, year, indicator, value
    return df[["country", "year", "indicator", "value"]]


def fetch_spending(start=2019, end=2025):
    """Military expenditure as % of GDP."""
    return fetch_indicator("MS.MIL.XPND.GD.ZS", "mil_pct_gdp", start, end)


def fetch_personnel(start=2019, end=2025):
    """Armed forces personnel, total headcount."""
    return fetch_indicator("MS.MIL.TOTL.P1", "personnel", start, end)


def create_schema(conn):
    """Create the indicators table with an explicit composite primary key.

    (country, year, indicator) together must be unique -- that's the
    definition of "one fact." value is deliberately NOT part of the key:
    it's the thing being identified, not part of the identifier.
    """
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
    print("Fetching data from World Bank...")
    spending_df = fetch_spending()
    personnel_df = fetch_personnel()

    # Stack (not merge!) the tidy tables on top of each other. Each source
    # contributes its own rows; nothing is joined side-by-side anymore.
    df = pd.concat([spending_df, personnel_df], ignore_index=True)

    print(f"Writing {len(df)} rows to {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    create_schema(conn)
    # if_exists="append" because create_schema() already made the table
    # (with its primary key) -- to_sql would otherwise create a plain
    # table with no key constraint at all.
    df.to_sql("indicators", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("Done. Database built.")

    # Sanity check: how many (country, year) pairs are missing at least
    # one indicator? In long format this means "a row is absent" rather
    # than "NaN in a column," so we check by counting indicators present
    # per country-year.
    counts = df.groupby(["country", "year"])["indicator"].nunique()
    incomplete = (counts < 2).sum()
    print(f"Country-years missing at least one indicator: {incomplete} / {len(counts)}")


if __name__ == "__main__":
    build_database()
    