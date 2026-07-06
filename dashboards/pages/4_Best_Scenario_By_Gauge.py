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

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Best Scenario by Gauge",
    page_icon="🏆",
    layout="wide",
)

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Change these filenames if yours are different
METRICS_PATH = PROCESSED_DIR / "best_scenario_by_gauge.csv"
GAUGE_LOCATION_PATH = PROCESSED_DIR / "gauge_locations.csv"
BOUNDARY_PATH = PROCESSED_DIR / "salt_verde_boundary.geojson"


@st.cache_data
def load_best_scenario_data() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        st.error(f"Missing file: {METRICS_PATH}")
        st.stop()

    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_gauge_locations() -> pd.DataFrame:
    if not GAUGE_LOCATION_PATH.exists():
        st.error(f"Missing file: {GAUGE_LOCATION_PATH}")
        st.stop()

    return pd.read_csv(GAUGE_LOCATION_PATH)


@st.cache_data
def load_boundary_geojson():
    if not BOUNDARY_PATH.exists():
        st.warning(f"Salt–Verde boundary file not found: {BOUNDARY_PATH}")
        return None

    gdf = gpd.read_file(BOUNDARY_PATH)
    gdf = gdf.to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


st.title("🏆 Best Scenario by Gauge Based on GNN KGESS")

st.markdown(
    """
    This page shows the best-performing GNN scenario at each USGS gauge based on
    the highest GNN KGESS value. Gauge colors represent the best GNN KGESS score.
    """
)

metrics = load_best_scenario_data()
locations = load_gauge_locations()
boundary_geojson = load_boundary_geojson()

required_metrics_cols = {"gauge_id", "best_scenario", "gnn_kgess"}
required_location_cols = {"gauge_id", "latitude", "longitude"}

missing_metrics = required_metrics_cols - set(metrics.columns)
missing_locations = required_location_cols - set(locations.columns)

if missing_metrics:
    st.error(f"Missing columns in {METRICS_PATH.name}: {missing_metrics}")
    st.write("Available columns:", list(metrics.columns))
    st.stop()

if missing_locations:
    st.error(f"Missing columns in {GAUGE_LOCATION_PATH.name}: {missing_locations}")
    st.write("Available columns:", list(locations.columns))
    st.stop()

df = metrics.merge(locations, on="gauge_id", how="left")

if df[["latitude", "longitude"]].isna().any().any():
    st.warning("Some gauges are missing latitude/longitude after merging.")

df = df.dropna(subset=["latitude", "longitude"])

if df.empty:
    st.error("No gauge data available for plotting after merging metrics and locations.")
    st.stop()

if "improvement" not in df.columns:
    if "rapid_kgess" in df.columns:
        df["improvement"] = df["gnn_kgess"] - df["rapid_kgess"]
    else:
        df["improvement"] = pd.NA

# --------------------------------------------------
# KPI cards
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Number of gauges", f"{df['gauge_id'].nunique()}")

with col2:
    st.metric("Mean best GNN KGESS", f"{df['gnn_kgess'].mean():.3f}")

with col3:
    if "rapid_kgess" in df.columns:
        st.metric("Mean RAPID KGESS", f"{df['rapid_kgess'].mean():.3f}")
    else:
        st.metric("Mean RAPID KGESS", "N/A")

with col4:
    if df["improvement"].notna().any():
        st.metric("Mean improvement", f"{df['improvement'].mean():.3f}")
    else:
        st.metric("Mean improvement", "N/A")

st.divider()

# --------------------------------------------------
# Map
# --------------------------------------------------
st.subheader("Map: Best GNN KGESS by Gauge")

center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    color="gnn_kgess",
    size="gnn_kgess",
    hover_name="gauge_id",
    hover_data={
        "best_scenario": True,
        "gnn_kgess": ":.3f",
        "rapid_kgess": ":.3f" if "rapid_kgess" in df.columns else False,
        "improvement": ":.3f" if df["improvement"].notna().any() else False,
        "latitude": False,
        "longitude": False,
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

st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# Bar chart
# --------------------------------------------------
st.subheader("Graph: Ranked Gauges by Best GNN KGESS")

ranked = df.sort_values("gnn_kgess", ascending=False)

bar_fig = px.bar(
    ranked,
    x="gnn_kgess",
    y="gauge_id",
    color="best_scenario",
    orientation="h",
    hover_data={
        "best_scenario": True,
        "gnn_kgess": ":.3f",
        "rapid_kgess": ":.3f" if "rapid_kgess" in ranked.columns else False,
        "improvement": ":.3f" if ranked["improvement"].notna().any() else False,
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
# Table
# --------------------------------------------------
st.subheader("Best Scenario Table")

display_cols = ["gauge_id", "best_scenario", "gnn_kgess"]

if "rapid_kgess" in df.columns:
    display_cols.append("rapid_kgess")

if "improvement" in df.columns:
    display_cols.append("improvement")

display_cols += ["latitude", "longitude"]

st.dataframe(
    ranked[display_cols],
    width="stretch",
    hide_index=True,
)