from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="LSTM-GNN Routing Dashboard",
    layout="wide",
)

st.title("LSTM-GNN Routing Dashboard")

st.markdown(
    """
    This dashboard summarizes GNN routing experiments, RAPID benchmarks,
    and NWM comparisons for the Salt-Verde river network.

    Use the pages in the sidebar to explore:

    - Scenario summaries
    - Gauge-level metrics
    - Spatial maps
    - Best scenario per gauge
    - GNN/RAPID/NWM comparisons
    """
)