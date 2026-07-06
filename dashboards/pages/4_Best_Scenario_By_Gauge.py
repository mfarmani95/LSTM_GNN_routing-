from __future__ import annotations

from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st

DB_PATH = Path("data/processed/routing_results.duckdb")

st.set_page_config(page_title="Best Scenario by Gauge", layout="wide")
st.title("Best Scenario by Gauge")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

con = duckdb.connect(str(DB_PATH), read_only=True)

df = con.sql("""
SELECT *
FROM best_scenario_by_gauge
ORDER BY gauge_id
""").fetchdf()

st.dataframe(df, use_container_width=True)

map_df = df.dropna(subset=["lat", "lon"]).copy()

fig = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    color="scenario_id",
    hover_name="gauge_id",
    hover_data=[
        "loss_type",
        "lag_days",
        "architecture",
        "gnn_kgess",
        "rapid_kgess",
        "nwm_kgess",
        "kgess_improvement_rapid",
    ],
    zoom=6,
    height=700,
    title="Best Scenario by Gauge Based on GNN KGEss",
)

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)

con.close()