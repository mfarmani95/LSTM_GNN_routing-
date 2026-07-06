from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import shapely.wkt
import streamlit as st


DB_PATH = Path("data/processed/routing_results.duckdb")

st.set_page_config(page_title="Spatial Map", layout="wide")
st.title("Spatial Map: Basin, Routing Graph, and Gauges")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()


@st.cache_data
def load_metrics(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.sql(
            """
            SELECT
                m.gauge_id,
                gl.lat,
                gl.lon,
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
            LEFT JOIN gauge_locations_combined gl
                ON m.gauge_id = gl.gauge_id
            WHERE gl.lat IS NOT NULL
              AND gl.lon IS NOT NULL
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data
def load_routing_edges(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.sql(
            """
            SELECT
                edge_id,
                from_lat,
                from_lon,
                to_lat,
                to_lon
            FROM routing_edges
            WHERE from_lat IS NOT NULL
              AND from_lon IS NOT NULL
              AND to_lat IS NOT NULL
              AND to_lon IS NOT NULL
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data
def load_basin_boundary(db_path: str) -> gpd.GeoDataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.sql(
            """
            SELECT
                feature_id,
                name,
                geometry_wkt
            FROM basin_boundary
            """
        ).fetchdf()
    finally:
        con.close()

    df["geometry"] = df["geometry_wkt"].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkt"]),
        geometry="geometry",
        crs="EPSG:4326",
    )


metrics_df = load_metrics(str(DB_PATH))

scenario_options = sorted(metrics_df["scenario_id"].unique())

selected_scenario = st.sidebar.selectbox(
    "Scenario",
    scenario_options,
)

metric = st.sidebar.selectbox(
    "Metric to map",
    [
        "gnn_kgess",
        "rapid_kgess",
        "nwm_kgess",
        "kgess_improvement_rapid",
        "kgess_improvement_nwm",
    ],
)

show_basin = st.sidebar.checkbox("Show basin boundary", value=True)
show_graph = st.sidebar.checkbox("Show routing graph", value=True)
show_gauges = st.sidebar.checkbox("Show gauges", value=True)

map_df = metrics_df[metrics_df["scenario_id"] == selected_scenario].copy()

fig = go.Figure()

if show_basin:
    basin_gdf = load_basin_boundary(str(DB_PATH))

    for _, row in basin_gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = []

        for poly in polygons:
            x, y = poly.exterior.xy
            fig.add_trace(
                go.Scattermapbox(
                    lon=list(x),
                    lat=list(y),
                    mode="lines",
                    line=dict(width=2, color="black"),
                    name="Basin boundary",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

if show_graph:
    edges = load_routing_edges(str(DB_PATH))

    max_edges = st.sidebar.number_input(
        "Max graph edges to draw",
        min_value=100,
        max_value=max(100, len(edges)),
        value=min(5000, len(edges)),
        step=100,
    )

    edges_plot = edges.head(int(max_edges))

    edge_lons = []
    edge_lats = []

    for row in edges_plot.itertuples(index=False):
        edge_lons.extend([row.from_lon, row.to_lon, None])
        edge_lats.extend([row.from_lat, row.to_lat, None])

    fig.add_trace(
        go.Scattermapbox(
            lon=edge_lons,
            lat=edge_lats,
            mode="lines",
            line=dict(width=1, color="gray"),
            name="Routing graph",
            hoverinfo="skip",
        )
    )

if show_gauges:
    fig.add_trace(
        go.Scattermapbox(
            lon=map_df["lon"],
            lat=map_df["lat"],
            mode="markers",
            marker=dict(
                size=14,
                color=map_df[metric],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=metric),
            ),
            text=map_df["gauge_id"],
            customdata=map_df[
                [
                    "scenario_id",
                    "gnn_kgess",
                    "rapid_kgess",
                    "nwm_kgess",
                    "kgess_improvement_rapid",
                    "kgess_improvement_nwm",
                ]
            ],
            hovertemplate=(
                "Gauge: %{text}<br>"
                "Scenario: %{customdata[0]}<br>"
                "GNN KGEss: %{customdata[1]:.3f}<br>"
                "RAPID KGEss: %{customdata[2]:.3f}<br>"
                "NWM KGEss: %{customdata[3]:.3f}<br>"
                "Improvement vs RAPID: %{customdata[4]:.3f}<br>"
                "Improvement vs NWM: %{customdata[5]:.3f}<extra></extra>"
            ),
            name="Gauges",
        )
    )

center_lat = float(map_df["lat"].mean())
center_lon = float(map_df["lon"].mean())

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=6,
    ),
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
    title=f"{metric} for {selected_scenario}",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Gauge metrics shown on map")
st.dataframe(
    map_df[
        [
            "gauge_id",
            "scenario_id",
            "loss_type",
            "lag_days",
            "gnn_kgess",
            "rapid_kgess",
            "nwm_kgess",
            "kgess_improvement_rapid",
            "kgess_improvement_nwm",
        ]
    ].sort_values("gnn_kgess", ascending=False),
    use_container_width=True,
)