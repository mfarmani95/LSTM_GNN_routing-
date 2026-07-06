from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


DB_PATH = Path("data/processed/routing_results.duckdb")


def check_database() -> None:
    if not DB_PATH.exists():
        st.error(
            f"Database not found: `{DB_PATH}`. "
            "Build it first or make sure the DuckDB file exists in this path."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_scenario_summary() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.sql(
            """
            SELECT *
            FROM scenario_summary
            ORDER BY mean_gnn_kgess DESC
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_gauge_metrics() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
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
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_best_scenario_by_gauge() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.sql(
            """
            SELECT
                b.gauge_id,
                gl.lat,
                gl.lon,
                b.scenario_id,
                b.label,
                b.loss_type,
                b.lag_days,
                b.architecture,
                b.gnn_kgess,
                b.rapid_kgess,
                b.nwm_kgess,
                b.kgess_improvement_rapid,
                b.kgess_improvement_nwm
            FROM best_scenario_by_gauge b
            LEFT JOIN gauge_locations_combined gl
                ON b.gauge_id = gl.gauge_id
            ORDER BY b.gauge_id
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_routing_edges() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
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


@st.cache_data(show_spinner=False)
def load_basin_boundary() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.sql(
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


@st.cache_data(show_spinner=False)
def load_graph_counts() -> pd.DataFrame:
    check_database()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM routing_nodes) AS n_nodes,
                (SELECT COUNT(*) FROM routing_edges) AS n_edges,
                (SELECT COUNT(*) FROM routing_gauges) AS n_graph_gauges,
                (SELECT COUNT(*) FROM runoff_mapping) AS n_runoff_mappings,
                (SELECT COUNT(*) FROM gauge_metrics) AS n_metric_rows
            """
        ).fetchdf()
    finally:
        con.close()