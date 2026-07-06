from __future__ import annotations

from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/processed/routing_results.duckdb")

st.set_page_config(page_title="Scenario Summary", layout="wide")
st.title("Scenario Summary")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

con = duckdb.connect(str(DB_PATH), read_only=True)

df = con.sql("""
SELECT *
FROM scenario_summary
ORDER BY mean_gnn_kgess DESC
""").fetchdf()

st.dataframe(df, use_container_width=True)

fig = px.bar(
    df,
    x="scenario_id",
    y="mean_gnn_kgess",
    color="loss_type",
    hover_data=[
        "lag_days",
        "architecture",
        "mean_improvement_vs_rapid",
        "mean_improvement_vs_nwm",
    ],
    title="Mean GNN KGEss by Scenario",
)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(
    df,
    x="scenario_id",
    y="mean_improvement_vs_rapid",
    color="loss_type",
    hover_data=["lag_days", "architecture"],
    title="Mean KGEss Improvement vs RAPID",
)

fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

con.close()