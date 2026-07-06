from pathlib import Path
import sys
import json

CURRENT_FILE = Path(__file__).resolve()

if CURRENT_FILE.parent.name == "pages":
    ROOT_DIR = CURRENT_FILE.parents[2]
else:
    ROOT_DIR = CURRENT_FILE.parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely import wkt


st.set_page_config(
    page_title="Best Scenario by Gauge",
    page_icon="🏆",
    layout="wide",
)

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "routing_results.duckdb"


@st.cache_data
def load_best_scenario_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Missing DuckDB file: {DB_PATH}")
        st.stop()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT
            gauge_id,
            lat,
            lon,
            scenario_id,
            label,
            loss_type,
            lag_days,
            architecture,
            gnn_kgess,
            rapid_kgess,
            nwm_kgess,
            kgess_improvement_rapid,
            kgess_improvement_nwm,
            rank
        FROM best_scenario_by_gauge
        WHERE rank = 1
    """

    df = con.execute(query).fetchdf()
    con.close()

    return df


@st.cache_data
def load_basin_boundary_geojson():
    if not DB_PATH.exists():
        return None

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        boundary_df = con.execute(
            """
            SELECT
                feature_id,
                name,
                geometry_wkt
            FROM basin_boundary
            """
        ).fetchdf()
    except Exception:
        con.close()
        return None

    con.close()

    if boundary_df.empty:
        return None

    try:
        boundary_df["geometry"] = boundary_df["geometry_wkt"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(
            boundary_df.drop(columns=["geometry_wkt"]),
            geometry="geometry",
            crs="EPSG:4326",
        )
        gdf = gdf.to_crs("EPSG:4326")
        return json.loads(gdf.to_json())
    except Exception:
        return None


st.title("🏆 Best Scenario by Gauge Based on GNN KGESS")

st.markdown(
    """
    This page shows the best-performing GNN scenario at each USGS gauge based on
    the highest GNN KGESS value. Gauge colors represent the best GNN KGESS score.
    """
)

df = load_best_scenario_data()
boundary_geojson = load_basin_boundary_geojson()

if df.empty:
    st.error("The `best_scenario_by_gauge` table returned no rows.")
    st.stop()

df["gauge_id"] = df["gauge_id"].astype(str)
df["gnn_kgess"] = pd.to_numeric(df["gnn_kgess"], errors="coerce")
df["rapid_kgess"] = pd.to_numeric(df["rapid_kgess"], errors="coerce")
df["nwm_kgess"] = pd.to_numeric(df["nwm_kgess"], errors="coerce")
df["kgess_improvement_rapid"] = pd.to_numeric(
    df["kgess_improvement_rapid"], errors="coerce"
)
df["kgess_improvement_nwm"] = pd.to_numeric(
    df["kgess_improvement_nwm"], errors="coerce"
)
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

df = df.dropna(subset=["gnn_kgess", "lat", "lon"])

if df.empty:
    st.error("No valid gauge rows after cleaning KGESS and coordinate columns.")
    st.stop()

# --------------------------------------------------
# KPI cards
# --------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Number of gauges", f"{df['gauge_id'].nunique()}")

with col2:
    st.metric("Mean best GNN KGESS", f"{df['gnn_kgess'].mean():.3f}")

with col3:
    st.metric("Mean RAPID KGESS", f"{df['rapid_kgess'].mean():.3f}")

with col4:
    st.metric("Mean NWM KGESS", f"{df['nwm_kgess'].mean():.3f}")

with col5:
    st.metric(
        "Mean improvement vs RAPID",
        f"{df['kgess_improvement_rapid'].mean():.3f}",
    )

st.divider()

# --------------------------------------------------
# Filters
# --------------------------------------------------
st.sidebar.header("Filters")

loss_options = sorted(df["loss_type"].dropna().unique().tolist())
selected_loss = st.sidebar.multiselect(
    "Loss type",
    options=loss_options,
    default=loss_options,
)

arch_options = sorted(df["architecture"].dropna().unique().tolist())
selected_arch = st.sidebar.multiselect(
    "Architecture",
    options=arch_options,
    default=arch_options,
)

plot_df = df[
    df["loss_type"].isin(selected_loss)
    & df["architecture"].isin(selected_arch)
].copy()

if plot_df.empty:
    st.warning("No gauges match the selected filters.")
    st.stop()

# --------------------------------------------------
# Map
# --------------------------------------------------
st.subheader("Map: Best GNN KGESS by Gauge")

center_lat = plot_df["lat"].mean()
center_lon = plot_df["lon"].mean()

fig = px.scatter_map(
    plot_df,
    lat="lat",
    lon="lon",
    color="gnn_kgess",
    size="gnn_kgess",
    hover_name="gauge_id",
    hover_data={
        "scenario_id": True,
        "label": True,
        "loss_type": True,
        "lag_days": True,
        "architecture": True,
        "gnn_kgess": ":.3f",
        "rapid_kgess": ":.3f",
        "nwm_kgess": ":.3f",
        "kgess_improvement_rapid": ":.3f",
        "kgess_improvement_nwm": ":.3f",
        "lat": False,
        "lon": False,
    },
    color_continuous_scale="Viridis",
    zoom=7,
    height=650,
    title="Best scenario by gauge based on GNN KGESS",
)

fig.update_layout(
    map=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=7,
    ),
    margin=dict(l=0, r=0, t=50, b=0),
)

if boundary_geojson is not None:
    fig.update_layout(
        map_layers=[
            {
                "source": boundary_geojson,
                "type": "line",
                "color": "black",
                "line": {"width": 2},
            }
        ]
    )
else:
    st.info("Basin boundary layer was not available from `basin_boundary`.")

st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# Graph 1: Ranked gauges by GNN KGESS
# --------------------------------------------------
st.subheader("Graph: Ranked Gauges by Best GNN KGESS")

ranked = plot_df.sort_values("gnn_kgess", ascending=False)

bar_fig = px.bar(
    ranked,
    x="gnn_kgess",
    y="gauge_id",
    color="scenario_id",
    orientation="h",
    hover_data={
        "scenario_id": True,
        "label": True,
        "loss_type": True,
        "lag_days": True,
        "architecture": True,
        "gnn_kgess": ":.3f",
        "rapid_kgess": ":.3f",
        "nwm_kgess": ":.3f",
        "kgess_improvement_rapid": ":.3f",
        "kgess_improvement_nwm": ":.3f",
    },
    title="Best GNN KGESS by gauge",
)

bar_fig.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Best GNN KGESS",
    yaxis_title="Gauge ID",
    height=650,
)

st.plotly_chart(bar_fig, width="stretch")

# --------------------------------------------------
# Graph 2: Improvement over RAPID
# --------------------------------------------------
st.subheader("Graph: Best-Scenario Improvement over RAPID")

improvement_ranked = plot_df.sort_values(
    "kgess_improvement_rapid",
    ascending=False,
)

improvement_fig = px.bar(
    improvement_ranked,
    x="kgess_improvement_rapid",
    y="gauge_id",
    color="scenario_id",
    orientation="h",
    hover_data={
        "scenario_id": True,
        "label": True,
        "loss_type": True,
        "lag_days": True,
        "architecture": True,
        "gnn_kgess": ":.3f",
        "rapid_kgess": ":.3f",
        "kgess_improvement_rapid": ":.3f",
    },
    title="Best-scenario KGESS improvement over RAPID by gauge",
)

improvement_fig.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="GNN KGESS - RAPID KGESS",
    yaxis_title="Gauge ID",
    height=650,
)

st.plotly_chart(improvement_fig, width="stretch")

# --------------------------------------------------
# Graph 3: Scenario frequency
# --------------------------------------------------
st.subheader("Graph: How Often Each Scenario Is Best")

scenario_counts = (
    plot_df.groupby(["scenario_id", "loss_type", "architecture"], as_index=False)
    .size()
    .rename(columns={"size": "n_gauges"})
    .sort_values("n_gauges", ascending=False)
)

scenario_fig = px.bar(
    scenario_counts,
    x="n_gauges",
    y="scenario_id",
    color="loss_type",
    orientation="h",
    hover_data={
        "scenario_id": True,
        "loss_type": True,
        "architecture": True,
        "n_gauges": True,
    },
    title="Number of gauges where each scenario is the best",
)

scenario_fig.update_layout(
    yaxis=dict(autorange="reversed"),
    xaxis_title="Number of gauges",
    yaxis_title="Scenario ID",
    height=500,
)

st.plotly_chart(scenario_fig, width="stretch")

# --------------------------------------------------
# Graph 4: GNN vs benchmarks
# --------------------------------------------------
st.subheader("Graph: GNN KGESS Compared with RAPID and NWM")

comparison_long = plot_df[
    ["gauge_id", "gnn_kgess", "rapid_kgess", "nwm_kgess"]
].melt(
    id_vars="gauge_id",
    value_vars=["gnn_kgess", "rapid_kgess", "nwm_kgess"],
    var_name="model",
    value_name="kgess",
)

comparison_long["model"] = comparison_long["model"].map(
    {
        "gnn_kgess": "Best GNN",
        "rapid_kgess": "RAPID",
        "nwm_kgess": "NWM",
    }
)

comparison_fig = px.box(
    comparison_long,
    x="model",
    y="kgess",
    points="all",
    title="Distribution of KGESS across gauges",
)

comparison_fig.update_layout(
    xaxis_title="Model",
    yaxis_title="KGESS",
    height=550,
)

st.plotly_chart(comparison_fig, width="stretch")

# --------------------------------------------------
# Table
# --------------------------------------------------
st.subheader("Best Scenario Table")

display_cols = [
    "gauge_id",
    "scenario_id",
    "label",
    "loss_type",
    "lag_days",
    "architecture",
    "gnn_kgess",
    "rapid_kgess",
    "nwm_kgess",
    "kgess_improvement_rapid",
    "kgess_improvement_nwm",
    "lat",
    "lon",
]

st.dataframe(
    ranked[display_cols],
    width="stretch",
    hide_index=True,
)

with st.expander("Debug: DuckDB source"):
    st.write("DuckDB path:", str(DB_PATH))
    st.write("Rows loaded:", len(df))
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(), width="stretch")
