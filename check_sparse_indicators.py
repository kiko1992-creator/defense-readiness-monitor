"""Check whether personnel and gov_debt_pct_gdp gaps follow a pattern
(specific countries missing, or specific years missing) or are scattered."""

import sqlite3
import pandas as pd

conn = sqlite3.connect("defense.db")
df = pd.read_sql("SELECT * FROM indicators", conn)
conn.close()

for indicator in ["personnel", "gov_debt_pct_gdp"]:
    sub = df[df["indicator"] == indicator].dropna(subset=["value"])
    print(f"\n--- {indicator}: {len(sub)} real values ---")
    print("By year:")
    print(sub.groupby("year").size())
    print("By country:")
    print(sub.groupby("country").size())