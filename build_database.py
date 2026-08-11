"""
Build the defense database.

Fetches military-expenditure data from the World Bank and stores it in a
local SQLite database (defense.db). Run this once to populate the database;
the dashboard and models then read from it instead of re-fetching each time.
"""

import sqlite3
import pandas as pd
import wbgapi as wb


COUNTRIES = ["POL", "LTU", "LVA", "EST", "USA", "NOR", "GRC", "GBR",
             "FRA", "DEU", "ITA", "ESP", "BEL", "NLD", "CAN"]

DB_FILE = "defense.db"


def fetch_spending(start=2019, end=2025):
    """Pull military expenditure (% GDP) from the World Bank as a tidy table."""
    df = wb.data.DataFrame("MS.MIL.XPND.GD.ZS", COUNTRIES, range(start, end))
    # Reshape from wide (year columns) to long (one row per country-year)
    df = df.stack().reset_index()
    df.columns = ["country", "year", "mil_pct_gdp"]
    df["year"] = df["year"].str.replace("YR", "").astype(int)
    return df


def build_database():
    """Fetch the data and write it into a SQLite table called 'spending'."""
    print("Fetching data from World Bank...")
    df = fetch_spending()

    print(f"Writing {len(df)} rows to {DB_FILE}...")
    # Open a connection to the database file (creates it if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    # Write the dataframe into a table named 'spending'
    df.to_sql("spending", conn, if_exists="replace", index=False)
    conn.close()

    print("Done. Database built.")


if __name__ == "__main__":
    build_database()

    