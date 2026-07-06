from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboards.components.data_loader import load_gauge_metrics
from dashboards.components.kpis import metric_card
from dashboards.components.style import apply_global_style


st.set_page_config(page_title="Gauge Explorer", page_icon="📍", layout="wide")
apply_global_style()

st.title("Gauge Explorer")
st.markdown(
    '<div class="small-muted">Inspect performance at individual gauges across scenarios.</div>',
    unsafe_allow_html=True,
)

df = load_gauge_metrics()

gauge_options = sorted(df["gauge_id"].dropna().unique())

with st.sidebar:
    st.header("Gauge selection")
    selected_gauge = st.selectbox("Gauge ID", gauge_options)

gdf = df[df["gauge_id"] == selected_gauge].copy()
best = gdf.sort_values("gnn_kgess", ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Gauge", selected_gauge)

with col2:
    metric_card("Best scenario", str(best["scenario_id"]))

with col3:
    metric_card("Best GNN KGEss", f'{best["gnn_kgess"]:.3f}')

with col4:
    metric_card("Best gain vs RAPID", f'{best["kgess_improvement_rapid"]:.3f}')

st.write("")

fig = px.bar(
    gdf.sort_values("gnn_kgess", ascending=False),
    x="scenario_id",
    y="gnn_kgess",
    color="loss_type",
    hover_data=[
        "rapid_kgess",
        "nwm_kgess",
        "kgess_improvement_rapid",
        "kgess_improvement_nwm",
    ],
    title=f"GNN KGEss by scenario for gauge {selected_gauge}",
)
fig.update_layout(
    xaxis_title="Scenario",
    yaxis_title="GNN KGEss",
    xaxis_tickangle=-45,
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig2 = px.scatter(
        gdf,
        x="rapid_kgess",
        y="gnn_kgess",
        color="loss_type",
        hover_name="scenario_id",
        title="GNN vs RAPID at this gauge",
    )
    fig2.add_shape(
        type="line",
        x0=gdf["rapid_kgess"].min(),
        y0=gdf["rapid_kgess"].min(),
        x1=gdf["rapid_kgess"].max(),
        y1=gdf["rapid_kgess"].max(),
        line=dict(dash="dash"),
    )
    fig2.update_layout(
        xaxis_title="RAPID KGEss",
        yaxis_title="GNN KGEss",
        height=500,
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    fig3 = px.scatter(
        gdf,
        x="nwm_kgess",
        y="gnn_kgess",
        color="loss_type",
        hover_name="scenario_id",
        title="GNN vs NWM at this gauge",
    )
    fig3.update_layout(
        xaxis_title="NWM KGEss",
        yaxis_title="GNN KGEss",
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Gauge table")
st.dataframe(
    gdf.sort_values("gnn_kgess", ascending=False),
    use_container_width=True,
    hide_index=True,
)