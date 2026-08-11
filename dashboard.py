"""
Defense Data Platform — Dashboard

An interactive Streamlit dashboard that reads defense-spending data from the
SQLite database and lets the user explore it by year and threshold.
"""

import sqlite3
import pandas as pd
import streamlit as st

DB_FILE = "defense.db"


def load_data(year):
    """Read one year's spending data from the database, ranked."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT country, mil_pct_gdp FROM spending WHERE year = ? ORDER BY mil_pct_gdp DESC",
        conn, params=(year,)
    )
    conn.close()
    return df


# ── Page setup ──
st.set_page_config(page_title="Defense Data Platform", layout="wide")
st.title("Defense Spending Dashboard")
st.caption("Military expenditure as % of GDP — sourced from World Bank, stored in SQLite.")

# ── Controls (sidebar) ──
year = st.sidebar.slider("Year", 2019, 2024, 2024)
threshold = st.sidebar.slider("Highlight spending above (% GDP)", 0.0, 5.0, 2.0, 0.1)

# ── Load and display ──
df = load_data(year)

st.subheader(f"Military spending, {year}")

# Bar chart
st.bar_chart(df.set_index("country")["mil_pct_gdp"])

# Table, with countries above the threshold flagged
df["above_threshold"] = df["mil_pct_gdp"] > threshold
st.dataframe(df, use_container_width=True)

# A quick summary metric
n_above = int(df["above_threshold"].sum())
st.metric(f"Countries above {threshold}% GDP", n_above)