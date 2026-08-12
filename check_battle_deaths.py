"""One-off check: what do the raw battle_deaths values actually look like?"""

import sqlite3
import pandas as pd

conn = sqlite3.connect("defense.db")
df = pd.read_sql("SELECT * FROM indicators WHERE indicator = 'battle_deaths'", conn)
conn.close()

pd.set_option("display.max_rows", None)
print(df[["country", "year", "value"]].sort_values("value", ascending=False))
print("\nHow many rows have value == 0:", (df["value"] == 0).sum(), "out of", len(df))
print("How many distinct countries have any nonzero value:",
      df.loc[df["value"] != 0, "country"].nunique())