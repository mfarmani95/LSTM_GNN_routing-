from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("data/processed/routing_results.duckdb")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def get_scenario_summary(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    con = connect(db_path)
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


def get_best_scenario_by_gauge(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    con = connect(db_path)
    try:
        return con.sql(
            """
            SELECT *
            FROM best_scenario_by_gauge
            ORDER BY gauge_id
            """
        ).fetchdf()
    finally:
        con.close()


def get_gauge_metrics(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    con = connect(db_path)
    try:
        return con.sql(
            """
            SELECT
                m.gauge_id,
                g.lat,
                g.lon,
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
            LEFT JOIN gauges g
                ON m.gauge_id = g.gauge_id
            ORDER BY
                s.loss_type,
                s.lag_days,
                m.gauge_id
            """
        ).fetchdf()
    finally:
        con.close()
