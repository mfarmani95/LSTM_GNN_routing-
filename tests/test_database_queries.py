from pathlib import Path

import pytest

from lstm_gnn_routing.database.queries import (
    get_best_scenario_by_gauge,
    get_gauge_metrics,
    get_scenario_summary,
)


DB_PATH = Path("data/processed/routing_results.duckdb")


@pytest.mark.skipif(not DB_PATH.exists(), reason="Local DuckDB database is not available")
def test_scenario_summary_has_rows():
    df = get_scenario_summary(DB_PATH)
    assert len(df) > 0
    assert "scenario_id" in df.columns
    assert "mean_gnn_kgess" in df.columns


@pytest.mark.skipif(not DB_PATH.exists(), reason="Local DuckDB database is not available")
def test_gauge_metrics_has_expected_columns():
    df = get_gauge_metrics(DB_PATH)
    expected = {
        "gauge_id",
        "scenario_id",
        "gnn_kgess",
        "rapid_kgess",
        "nwm_kgess",
        "kgess_improvement_rapid",
        "kgess_improvement_nwm",
    }
    assert expected.issubset(df.columns)


@pytest.mark.skipif(not DB_PATH.exists(), reason="Local DuckDB database is not available")
def test_best_scenario_by_gauge_has_rows():
    df = get_best_scenario_by_gauge(DB_PATH)
    assert len(df) > 0
    assert "gauge_id" in df.columns
    assert "scenario_id" in df.columns
