from pathlib import Path

import duckdb
import pytest


DB_PATH = Path("data/processed/routing_results.duckdb")


@pytest.mark.skipif(not DB_PATH.exists(), reason="Local DuckDB database is not available")
def test_spatial_tables_exist():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = set(con.sql("SHOW TABLES").fetchdf()["name"])
    finally:
        con.close()

    expected = {
        "basin_boundary",
        "routing_nodes",
        "routing_edges",
        "routing_gauges",
        "gauge_locations_combined",
    }

    assert expected.issubset(tables)


@pytest.mark.skipif(not DB_PATH.exists(), reason="Local DuckDB database is not available")
def test_routing_graph_has_expected_size():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        result = con.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM routing_nodes) AS n_nodes,
                (SELECT COUNT(*) FROM routing_edges) AS n_edges
            """
        ).fetchone()
    finally:
        con.close()

    n_nodes, n_edges = result
    assert n_nodes > 0
    assert n_edges > 0