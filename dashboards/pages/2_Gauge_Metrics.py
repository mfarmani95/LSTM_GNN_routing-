from __future__ import annotations

from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/processed/routing_results.duckdb")

st.set_page_config(page_title="Gauge Metrics", layout="wide")
st.title("Gauge-Level Metrics")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

con = duckdb.connect(str(DB_PATH), read_only=True)

df = con.sql("""
SELECT
    m.gauge_id,
    g.lat,
    g.lon,
    m.scenario_id,
    s.label,
    s.loss_type,
    s.lag_days,
    s.architecture,
    m.gnn_kgess,
    m.rapid_kgess,
    m.nwm_kgess,
    m.kgess_improvement_rapid,
    m.kgess_improvement_nwm
FROM gauge_metrics m
JOIN scenarios s
    ON m.scenario_id = s.scenario_id
LEFT JOIN gauges g
    ON m.gauge_id = g.gauge_id
""").fetchdf()

loss_types = sorted(df["loss_type"].dropna().unique())
selected_losses = st.sidebar.multiselect(
    "Loss type",
    loss_types,
    default=loss_types,
)

filtered = df[df["loss_type"].isin(selected_losses)].copy()

st.dataframe(filtered, use_container_width=True)

fig = px.box(
    filtered,
    x="scenario_id",
    y="gnn_kgess",
    color="loss_type",
    points="all",
    title="GNN KGEss Distribution by Scenario",
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.box(
    filtered,
    x="scenario_id",
    y="kgess_improvement_rapid",
    color="loss_type",
    points="all",
    title="KGEss Improvement vs RAPID",
)
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

con.close()