from __future__ import annotations

from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st


DB_PATH = Path("data/processed/routing_results.duckdb")

st.set_page_config(page_title="Graph Summary", layout="wide")
st.title("Routing Graph Summary")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

con = duckdb.connect(str(DB_PATH), read_only=True)

counts = con.sql("""
SELECT
    (SELECT COUNT(*) FROM routing_nodes) AS n_nodes,
    (SELECT COUNT(*) FROM routing_edges) AS n_edges,
    (SELECT COUNT(*) FROM routing_gauges) AS n_graph_gauges,
    (SELECT COUNT(*) FROM runoff_mapping) AS n_runoff_mappings
""").fetchdf()

st.subheader("Graph counts")
st.dataframe(counts, use_container_width=True)

edge_feature_names = con.sql("""
SELECT DISTINCT feature_name
FROM routing_edge_features
ORDER BY feature_name
""").fetchdf()["feature_name"].tolist()

selected_edge_feature = st.selectbox(
    "Edge feature",
    edge_feature_names,
    index=edge_feature_names.index("Length_m") if "Length_m" in edge_feature_names else 0,
)

edge_df = con.sql(
    """
    SELECT
        edge_id,
        feature_value
    FROM routing_edge_features
    WHERE feature_name = ?
    """,
    params=[selected_edge_feature],
).fetchdf()

fig = px.histogram(
    edge_df,
    x="feature_value",
    nbins=60,
    title=f"Distribution of edge feature: {selected_edge_feature}",
)
st.plotly_chart(fig, use_container_width=True)

node_feature_names = con.sql("""
SELECT DISTINCT feature_name
FROM routing_node_features
ORDER BY feature_name
""").fetchdf()["feature_name"].tolist()

selected_node_feature = st.selectbox(
    "Node feature",
    node_feature_names,
    index=node_feature_names.index("distance_to_outlet_m")
    if "distance_to_outlet_m" in node_feature_names
    else 0,
)

node_df = con.sql(
    """
    SELECT
        node_id,
        feature_value
    FROM routing_node_features
    WHERE feature_name = ?
    """,
    params=[selected_node_feature],
).fetchdf()

fig2 = px.histogram(
    node_df,
    x="feature_value",
    nbins=60,
    title=f"Distribution of node feature: {selected_node_feature}",
)
st.plotly_chart(fig2, use_container_width=True)

con.close()