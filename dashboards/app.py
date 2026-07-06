from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("data/processed/routing_results.duckdb")


@st.cache_data
def load_metrics(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.sql(
            """
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
            """
        ).fetchdf()
    finally:
        con.close()


st.set_page_config(page_title="LSTM-GNN Routing Dashboard", layout="wide")

st.title("LSTM-GNN Routing Results Dashboard")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

df = load_metrics(str(DB_PATH))

loss_types = sorted(df["loss_type"].dropna().unique())
selected_losses = st.sidebar.multiselect(
    "Loss type",
    options=loss_types,
    default=loss_types,
)

filtered = df[df["loss_type"].isin(selected_losses)].copy()

st.subheader("Scenario Summary")

summary = (
    filtered.groupby(["scenario_id", "loss_type", "lag_days", "architecture"], dropna=False)
    .agg(
        n_gauges=("gauge_id", "nunique"),
        mean_gnn_kgess=("gnn_kgess", "mean"),
        median_gnn_kgess=("gnn_kgess", "median"),
        mean_improvement_vs_rapid=("kgess_improvement_rapid", "mean"),
        mean_improvement_vs_nwm=("kgess_improvement_nwm", "mean"),
    )
    .reset_index()
    .sort_values("mean_gnn_kgess", ascending=False)
)

st.dataframe(summary, use_container_width=True)

st.subheader("GNN KGEss by Scenario")

fig = px.box(
    filtered,
    x="scenario_id",
    y="gnn_kgess",
    color="loss_type",
    points="all",
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Improvement vs RAPID")

fig2 = px.box(
    filtered,
    x="scenario_id",
    y="kgess_improvement_rapid",
    color="loss_type",
    points="all",
)
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Gauge Map")

map_df = filtered.dropna(subset=["lat", "lon"]).copy()

if len(map_df) > 0:
    selected_scenario = st.selectbox(
        "Scenario for map",
        options=sorted(map_df["scenario_id"].unique()),
    )

    map_s = map_df[map_df["scenario_id"] == selected_scenario]

    fig3 = px.scatter_mapbox(
        map_s,
        lat="lat",
        lon="lon",
        color="gnn_kgess",
        hover_name="gauge_id",
        hover_data=[
            "scenario_id",
            "gnn_kgess",
            "rapid_kgess",
            "nwm_kgess",
            "kgess_improvement_rapid",
        ],
        zoom=6,
        height=600,
    )
    fig3.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No latitude/longitude columns available for mapping.")
