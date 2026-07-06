from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from lstm_gnn_routing.database.scenario_registry import (
    ORDERED_SCENARIOS,
    SCENARIO_BEST_MODELS,
    SCENARIO_LABELS,
    infer_architecture,
    infer_lag_days,
    infer_loss_type,
)


def build_scenario_table(metric_file_name: str) -> pd.DataFrame:
    rows = []

    for scenario_name in ORDERED_SCENARIOS:
        if scenario_name not in SCENARIO_BEST_MODELS:
            print(f"Missing scenario in registry: {scenario_name}")
            continue

        info = SCENARIO_BEST_MODELS[scenario_name]
        base_dir = Path(info["base_dir"])
        seed = info["seed"]
        eval_dir = base_dir / seed / "evaluation_test_best_final_stage_model"

        rows.append(
            {
                "scenario_id": scenario_name,
                "scenario_name": scenario_name,
                "label": SCENARIO_LABELS.get(scenario_name, scenario_name),
                "loss_type": infer_loss_type(scenario_name),
                "lag_days": infer_lag_days(scenario_name),
                "architecture": infer_architecture(scenario_name),
                "base_dir": str(base_dir),
                "seed": seed,
                "eval_dir": str(eval_dir),
                "rapid_gnn_metrics_path": str(
                    eval_dir / "rapid_comparison" / "rapid_gnn_metrics.csv"
                ),
                "nwm_metrics_path": str(eval_dir / metric_file_name),
            }
        )

    return pd.DataFrame(rows)


def build_metrics_table(
    scenario_df: pd.DataFrame,
    metric_file_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    gauge_rows = []

    for row in scenario_df.itertuples(index=False):
        scenario_id = row.scenario_id
        eval_dir = Path(row.eval_dir)

        rapid_gnn_path = eval_dir / "rapid_comparison" / "rapid_gnn_metrics.csv"
        nwm_path = eval_dir / metric_file_name

        if not rapid_gnn_path.is_file():
            print(f"Missing RAPID/GNN file: {rapid_gnn_path}")
            continue

        if not nwm_path.is_file():
            print(f"Missing NWM file: {nwm_path}")
            continue

        df_rg = pd.read_csv(rapid_gnn_path, dtype={"gauge_id": str})
        df_rg = df_rg[df_rg["scale"] == "daily"].copy()

        # Gauge metadata from GNN rows, if lat/lon exist
        if {"gauge_id", "lat", "lon"}.issubset(df_rg.columns):
            gdf = (
                df_rg[["gauge_id", "lat", "lon"]]
                .dropna(subset=["gauge_id"])
                .drop_duplicates(subset=["gauge_id"])
                .copy()
            )
            gauge_rows.append(gdf)

        gnn_df = df_rg[df_rg["model"] == "GNN"][["gauge_id", "kgess"]].rename(
            columns={"kgess": "gnn_kgess"}
        )

        rapid_df = df_rg[df_rg["model"] == "RAPID"][["gauge_id", "kgess"]].rename(
            columns={"kgess": "rapid_kgess"}
        )

        merged = pd.merge(gnn_df, rapid_df, on="gauge_id", how="inner")

        df_nwm = pd.read_csv(nwm_path, dtype={"gauge_id": str})

        if "nwm_kgess" not in df_nwm.columns:
            print(f"Missing nwm_kgess column in: {nwm_path}")
            continue

        nwm_df = df_nwm[["gauge_id", "nwm_kgess"]].copy()
        merged = pd.merge(merged, nwm_df, on="gauge_id", how="inner")

        merged["scenario_id"] = scenario_id
        merged["scale"] = "daily"

        merged["kgess_improvement_rapid"] = (
            merged["gnn_kgess"] - merged["rapid_kgess"]
        )
        merged["kgess_improvement_nwm"] = merged["gnn_kgess"] - merged["nwm_kgess"]

        metric_rows.append(
            merged[
                [
                    "scenario_id",
                    "gauge_id",
                    "scale",
                    "gnn_kgess",
                    "rapid_kgess",
                    "nwm_kgess",
                    "kgess_improvement_rapid",
                    "kgess_improvement_nwm",
                ]
            ]
        )

    if not metric_rows:
        raise RuntimeError("No metric files were loaded.")

    metrics_df = pd.concat(metric_rows, ignore_index=True)

    if gauge_rows:
        gauges_df = (
            pd.concat(gauge_rows, ignore_index=True)
            .drop_duplicates(subset=["gauge_id"])
            .sort_values("gauge_id")
            .reset_index(drop=True)
        )
    else:
        gauges_df = pd.DataFrame(columns=["gauge_id", "lat", "lon"])

    return metrics_df, gauges_df


def write_database(
    db_path: Path,
    scenarios_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    gauges_df: pd.DataFrame,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    con.execute("DROP TABLE IF EXISTS scenarios")
    con.execute("DROP TABLE IF EXISTS gauge_metrics")
    con.execute("DROP TABLE IF EXISTS gauges")

    con.register("scenarios_df", scenarios_df)
    con.register("metrics_df", metrics_df)
    con.register("gauges_df", gauges_df)

    con.execute("CREATE TABLE scenarios AS SELECT * FROM scenarios_df")
    con.execute("CREATE TABLE gauge_metrics AS SELECT * FROM metrics_df")
    con.execute("CREATE TABLE gauges AS SELECT * FROM gauges_df")

    con.execute(
        """
        CREATE OR REPLACE VIEW scenario_summary AS
        SELECT
            s.scenario_id,
            s.label,
            s.loss_type,
            s.lag_days,
            s.architecture,
            COUNT(*) AS n_gauges,
            AVG(m.gnn_kgess) AS mean_gnn_kgess,
            MEDIAN(m.gnn_kgess) AS median_gnn_kgess,
            AVG(m.rapid_kgess) AS mean_rapid_kgess,
            AVG(m.nwm_kgess) AS mean_nwm_kgess,
            AVG(m.kgess_improvement_rapid) AS mean_improvement_vs_rapid,
            AVG(m.kgess_improvement_nwm) AS mean_improvement_vs_nwm
        FROM gauge_metrics m
        JOIN scenarios s
            ON m.scenario_id = s.scenario_id
        GROUP BY
            s.scenario_id,
            s.label,
            s.loss_type,
            s.lag_days,
            s.architecture
        ORDER BY
            s.loss_type,
            s.lag_days
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW best_scenario_by_gauge AS
        SELECT *
        FROM (
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
                m.kgess_improvement_nwm,
                ROW_NUMBER() OVER (
                    PARTITION BY m.gauge_id
                    ORDER BY m.gnn_kgess DESC
                ) AS rank
            FROM gauge_metrics m
            JOIN scenarios s
                ON m.scenario_id = s.scenario_id
            LEFT JOIN gauges g
                ON m.gauge_id = g.gauge_id
        )
        WHERE rank = 1
        """
    )

    print(f"Database written to: {db_path}")
    print(con.execute("SHOW TABLES").fetchdf())
    print(con.execute("SELECT * FROM scenario_summary").fetchdf())

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/processed/routing_results.duckdb"),
    )
    parser.add_argument(
        "--metric-file-name",
        type=str,
        default="test_metrics_simulation_nwm_rapid_by_gauge.csv",
    )

    args = parser.parse_args()

    scenarios_df = build_scenario_table(metric_file_name=args.metric_file_name)
    metrics_df, gauges_df = build_metrics_table(
        scenario_df=scenarios_df,
        metric_file_name=args.metric_file_name,
    )

    write_database(
        db_path=args.db_path,
        scenarios_df=scenarios_df,
        metrics_df=metrics_df,
        gauges_df=gauges_df,
    )


if __name__ == "__main__":
    main()