"""One-off check: how many rows does each indicator actually have?"""

import sqlite3
import pandas as pd

conn = sqlite3.connect("defense.db")
df = pd.read_sql("SELECT * FROM indicators", conn)
conn.close()

print(df.groupby("indicator").size())