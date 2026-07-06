from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboards.components.data_loader import (
    load_best_scenario_by_gauge,
    load_gauge_metrics,
    load_scenario_summary,
)
from dashboards.components.kpis import metric_card
from dashboards.components.style import apply_global_style


st.set_page_config(page_title="Executive Summary", page_icon="📊", layout="wide")
apply_global_style()

st.title("Executive Summary")
st.markdown(
    '<div class="small-muted">High-level ranking of routing scenarios and benchmark improvements.</div>',
    unsafe_allow_html=True,
)

summary = load_scenario_summary()
metrics = load_gauge_metrics()
best_by_gauge = load_best_scenario_by_gauge()

best = summary.iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Top scenario", str(best["scenario_id"]), "Ranked by mean GNN KGEss")

with col2:
    metric_card("Top mean GNN KGEss", f'{best["mean_gnn_kgess"]:.3f}')

with col3:
    metric_card("Mean gain vs RAPID", f'{best["mean_improvement_vs_rapid"]:.3f}')

with col4:
    metric_card("Mean gain vs NWM", f'{best["mean_improvement_vs_nwm"]:.3f}')

st.write("")

left, right = st.columns([1.2, 1])

with left:
    fig = px.bar(
        summary,
        x="scenario_id",
        y="mean_gnn_kgess",
        color="loss_type",
        hover_data=[
            "lag_days",
            "architecture",
            "mean_improvement_vs_rapid",
            "mean_improvement_vs_nwm",
        ],
        title="Mean GNN KGEss by scenario",
    )
    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Mean GNN KGEss",
        xaxis_tickangle=-45,
        legend_title="Loss",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig2 = px.bar(
        summary,
        x="scenario_id",
        y="mean_improvement_vs_rapid",
        color="loss_type",
        hover_data=["lag_days", "architecture"],
        title="Mean improvement vs RAPID",
    )
    fig2.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Mean KGEss improvement",
        xaxis_tickangle=-45,
        legend_title="Loss",
        height=520,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Best scenario by gauge")

counts = (
    best_by_gauge.groupby("scenario_id")
    .size()
    .reset_index(name="n_gauges")
    .sort_values("n_gauges", ascending=False)
)

fig3 = px.bar(
    counts,
    x="scenario_id",
    y="n_gauges",
    title="Number of gauges where each scenario performs best",
)
fig3.update_layout(
    xaxis_title="Scenario",
    yaxis_title="Gauge count",
    xaxis_tickangle=-45,
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("View scenario summary table"):
    st.dataframe(summary, use_container_width=True, hide_index=True)

with st.expander("View best scenario by gauge"):
    st.dataframe(best_by_gauge, use_container_width=True, hide_index=True)