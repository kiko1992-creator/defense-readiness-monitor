"""
Query the defense database.

Demonstrates reading data back out of the SQLite database using SQL —
the language for asking a database questions.
"""

import sqlite3
import pandas as pd

DB_FILE = "defense.db"


def run_query(sql):
    """Run a SQL query against the database and return the result as a table."""
    conn = sqlite3.connect(DB_FILE)
    result = pd.read_sql_query(sql, conn)
    conn.close()
    return result


if __name__ == "__main__":
    # 1. How many rows are in the database?
    print("Total rows stored:")
    print(run_query("SELECT COUNT(*) AS rows FROM spending"))

    # 2. Show the 2024 spending, highest first
    print("\n2024 military spending (% GDP), highest first:")
    print(run_query("""
        SELECT country, mil_pct_gdp
        FROM spending
        WHERE year = 2024
        ORDER BY mil_pct_gdp DESC
    """).to_string(index=False))

    # 3. Only the big spenders — above 3% of GDP in 2024
    print("\nCountries spending above 3% of GDP in 2024:")
    print(run_query("""
        SELECT country, mil_pct_gdp
        FROM spending
        WHERE year = 2024 AND mil_pct_gdp > 3.0
        ORDER BY mil_pct_gdp DESC
    """).to_string(index=False))
    