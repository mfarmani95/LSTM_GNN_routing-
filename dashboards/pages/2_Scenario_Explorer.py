from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboards.components.data_loader import load_gauge_metrics
from dashboards.components.kpis import metric_card
from dashboards.components.style import apply_global_style


st.set_page_config(page_title="Scenario Explorer", page_icon="🧪", layout="wide")
apply_global_style()

st.title("Scenario Explorer")
st.markdown(
    '<div class="small-muted">Compare GNN routing performance across lag windows, loss functions, and architectures.</div>',
    unsafe_allow_html=True,
)

df = load_gauge_metrics()

loss_options = sorted(df["loss_type"].dropna().unique())
arch_options = sorted(df["architecture"].dropna().unique())
scenario_options = sorted(df["scenario_id"].dropna().unique())

with st.sidebar:
    st.header("Filters")
    selected_losses = st.multiselect("Loss type", loss_options, default=loss_options)
    selected_arch = st.multiselect("Architecture", arch_options, default=arch_options)
    selected_scenarios = st.multiselect(
        "Scenarios",
        scenario_options,
        default=scenario_options,
    )

filtered = df[
    df["loss_type"].isin(selected_losses)
    & df["architecture"].isin(selected_arch)
    & df["scenario_id"].isin(selected_scenarios)
].copy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Scenarios", f'{filtered["scenario_id"].nunique():,}')

with col2:
    metric_card("Gauges", f'{filtered["gauge_id"].nunique():,}')

with col3:
    metric_card("Mean GNN KGEss", f'{filtered["gnn_kgess"].mean():.3f}')

with col4:
    metric_card("Mean gain vs RAPID", f'{filtered["kgess_improvement_rapid"].mean():.3f}')

st.write("")

fig = px.box(
    filtered,
    x="scenario_id",
    y="gnn_kgess",
    color="loss_type",
    points="all",
    title="GNN KGEss distribution by scenario",
)
fig.update_layout(
    xaxis_title="Scenario",
    yaxis_title="GNN KGEss",
    xaxis_tickangle=-45,
    height=560,
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig2 = px.box(
        filtered,
        x="scenario_id",
        y="kgess_improvement_rapid",
        color="loss_type",
        points="all",
        title="Improvement over RAPID",
    )
    fig2.update_layout(
        xaxis_title="Scenario",
        yaxis_title="GNN KGEss − RAPID KGEss",
        xaxis_tickangle=-45,
        height=500,
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    fig3 = px.box(
        filtered,
        x="scenario_id",
        y="kgess_improvement_nwm",
        color="loss_type",
        points="all",
        title="Improvement over NWM",
    )
    fig3.update_layout(
        xaxis_title="Scenario",
        yaxis_title="GNN KGEss − NWM KGEss",
        xaxis_tickangle=-45,
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("View filtered gauge-level data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)