from __future__ import annotations

import streamlit as st

from dashboards.components.data_loader import (
    check_database,
    load_graph_counts,
    load_scenario_summary,
)
from dashboards.components.kpis import metric_card
from dashboards.components.style import apply_global_style


st.set_page_config(
    page_title="LSTM-GNN Routing Dashboard",
    page_icon="🌊",
    layout="wide",
)

apply_global_style()
check_database()

summary = load_scenario_summary()
counts = load_graph_counts().iloc[0]

best = summary.iloc[0]

st.title("LSTM-GNN Routing Dashboard")

st.markdown(
    """
    <div class="small-muted">
    Interactive analytics dashboard for graph neural network streamflow routing experiments,
    RAPID and NWM benchmark comparison, and Salt–Verde river-network diagnostics.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "Best scenario",
        str(best["scenario_id"]),
        "Highest mean GNN KGEss",
    )

with col2:
    metric_card(
        "Mean GNN KGEss",
        f'{best["mean_gnn_kgess"]:.3f}',
        "Best scenario average",
    )

with col3:
    metric_card(
        "Routing graph",
        f'{int(counts["n_nodes"]):,} nodes',
        f'{int(counts["n_edges"]):,} edges',
    )

with col4:
    metric_card(
        "Gauge metrics",
        f'{int(counts["n_metric_rows"]):,} rows',
        "Scenario × gauge records",
    )

st.write("")

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Project overview")
    st.markdown(
        """
        This dashboard summarizes the performance of LSTM/GNN-based runoff routing
        experiments over the Salt–Verde river network.

        The database combines:

        - GNN streamflow-routing metrics by gauge and scenario
        - RAPID benchmark metrics
        - NWM comparison metrics
        - Routing graph topology and hydrologic edge/node features
        - Gauge locations and basin boundary geometry

        Use the sidebar pages to move from high-level scenario ranking to
        gauge-level diagnostics and spatial network exploration.
        """
    )

with right:
    st.subheader("Best scenarios")
    display_cols = [
        "scenario_id",
        "loss_type",
        "lag_days",
        "architecture",
        "mean_gnn_kgess",
        "mean_improvement_vs_rapid",
    ]
    st.dataframe(
        summary[display_cols].head(8),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Recommended workflow")
st.markdown(
    """
    1. Start with **Executive Summary** to identify the best scenarios.
    2. Use **Scenario Explorer** to compare loss functions and lag windows.
    3. Use **Gauge Explorer** to inspect gauge-level performance.
    4. Use **Spatial Network Map** to view gauges, basin boundary, and routing graph.
    5. Use **Graph Diagnostics** to inspect node and edge feature distributions.
    """
)