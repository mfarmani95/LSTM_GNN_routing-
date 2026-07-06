from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr


DEFAULT_DB_PATH = Path("data/processed/routing_results.duckdb")
DEFAULT_BASIN_SHP = Path("data/HUC4/Salt_and_Verde.shp")
DEFAULT_GRAPH_NC = Path(
    "data/graphs/routing_graph_ngen_salt_verde_center_mapping_derived_hydrology.nc"
)


def resolve_path(path_text: str | None, project_root: Path) -> Path | None:
    if not path_text:
        return None

    path = Path(path_text)
    if path.is_absolute():
        return path

    return project_root / path


def build_basin_boundary_table(shp_path: Path) -> pd.DataFrame:
    if not shp_path.exists():
        raise FileNotFoundError(f"Basin shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        print("Warning: basin shapefile has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.reset_index(drop=True)

    name_col = None
    for candidate in ["name", "Name", "NAME", "huc4", "HUC4", "id", "ID"]:
        if candidate in gdf.columns:
            name_col = candidate
            break

    rows = []
    for idx, row in gdf.iterrows():
        rows.append(
            {
                "feature_id": int(idx),
                "name": str(row[name_col]) if name_col is not None else "Salt_and_Verde",
                "geometry_wkt": row.geometry.wkt,
            }
        )

    return pd.DataFrame(rows)


def convert_grid_to_lonlat(
    node_y: np.ndarray,
    node_x: np.ndarray,
    dem_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    lon = np.full(len(node_y), np.nan, dtype=float)
    lat = np.full(len(node_y), np.nan, dtype=float)

    if dem_path is None or not dem_path.exists():
        print(f"Warning: DEM not found: {dem_path}. Nodes will not have lon/lat.")
        return lon, lat

    with rasterio.open(dem_path) as src:
        xs, ys = rasterio.transform.xy(
            src.transform,
            node_y.astype(int),
            node_x.astype(int),
            offset="center",
        )

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if src.crs is None:
            print("Warning: DEM has no CRS. Assuming x/y are already lon/lat.")
            return xs, ys

        if src.crs.to_epsg() == 4326:
            return xs, ys

        transformer = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs, ys)

    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def build_graph_tables(
    graph_nc_path: Path,
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    if not graph_nc_path.exists():
        raise FileNotFoundError(f"Routing graph NetCDF not found: {graph_nc_path}")

    ds = xr.open_dataset(graph_nc_path)

    edge_source = ds["edge_source"].values.astype(int)
    edge_target = ds["edge_target"].values.astype(int)
    node_y = ds["node_y"].values.astype(int)
    node_x = ds["node_x"].values.astype(int)
    flat_index = ds["flat_index"].values.astype(int)

    n_nodes = len(node_y)
    n_edges = len(edge_source)

    dem_attr = ds.attrs.get("dem")
    dem_path = resolve_path(dem_attr, project_root)

    lon, lat = convert_grid_to_lonlat(
        node_y=node_y,
        node_x=node_x,
        dem_path=dem_path,
    )

    nodes_df = pd.DataFrame(
        {
            "node_id": np.arange(n_nodes, dtype=int),
            "flat_index": flat_index,
            "node_y": node_y,
            "node_x": node_x,
            "lat": lat,
            "lon": lon,
        }
    )

    edges_df = pd.DataFrame(
        {
            "edge_id": np.arange(n_edges, dtype=int),
            "from_node": edge_source,
            "to_node": edge_target,
        }
    )

    edges_df = edges_df.merge(
        nodes_df[["node_id", "lat", "lon"]],
        left_on="from_node",
        right_on="node_id",
        how="left",
    ).rename(columns={"lat": "from_lat", "lon": "from_lon"})

    edges_df = edges_df.drop(columns=["node_id"])

    edges_df = edges_df.merge(
        nodes_df[["node_id", "lat", "lon"]],
        left_on="to_node",
        right_on="node_id",
        how="left",
    ).rename(columns={"lat": "to_lat", "lon": "to_lon"})

    edges_df = edges_df.drop(columns=["node_id"])

    edge_feature_names = [str(v) for v in ds["edge_feature"].values]
    edge_attr = ds["edge_attr"].values

    edge_feature_rows = []
    for edge_id in range(edge_attr.shape[0]):
        for feature_idx, feature_name in enumerate(edge_feature_names):
            value = edge_attr[edge_id, feature_idx]
            if np.isfinite(value):
                edge_feature_rows.append(
                    {
                        "edge_id": int(edge_id),
                        "feature_name": feature_name,
                        "feature_value": float(value),
                    }
                )

    edge_features_df = pd.DataFrame(
        edge_feature_rows,
        columns=["edge_id", "feature_name", "feature_value"],
    )

    node_feature_names = [str(v) for v in ds["node_feature"].values]
    node_features = ds["node_features"].values

    node_feature_rows = []
    for node_id in range(node_features.shape[0]):
        for feature_idx, feature_name in enumerate(node_feature_names):
            value = node_features[node_id, feature_idx]
            if np.isfinite(value):
                node_feature_rows.append(
                    {
                        "node_id": int(node_id),
                        "feature_name": feature_name,
                        "feature_value": float(value),
                    }
                )

    node_features_df = pd.DataFrame(
        node_feature_rows,
        columns=["node_id", "feature_name", "feature_value"],
    )

    gauge_index = ds["gauge_index"].values.astype(int)
    gauge_id = [str(v) for v in ds["gauge_id"].values]

    routing_gauges_df = pd.DataFrame(
        {
            "gauge_id": gauge_id,
            "node_id": gauge_index,
        }
    )

    routing_gauges_df = routing_gauges_df.merge(
        nodes_df[["node_id", "lat", "lon", "node_y", "node_x", "flat_index"]],
        on="node_id",
        how="left",
    )

    runoff_mapping_df = pd.DataFrame(
        {
            "runoff_source": ds["runoff_source"].values.astype(int),
            "runoff_target_index": ds["runoff_target_index"].values.astype(int),
            "runoff_source_index": ds["runoff_source_index"].values.astype(int),
            "runoff_source_flat_index": ds["runoff_source_flat_index"].values.astype(int),
            "runoff_source_weight": ds["runoff_source_weight"].values.astype(float),
            "runoff_source_fraction": ds["runoff_source_fraction"].values.astype(float),
        }
    )

    runoff_feature_names = [str(v) for v in ds["runoff_source_feature"].values]
    runoff_features = ds["runoff_source_features"].values

    runoff_feature_rows = []
    for source_id in range(runoff_features.shape[0]):
        for feature_idx, feature_name in enumerate(runoff_feature_names):
            value = runoff_features[source_id, feature_idx]
            if np.isfinite(value):
                runoff_feature_rows.append(
                    {
                        "runoff_source": int(source_id),
                        "feature_name": feature_name,
                        "feature_value": float(value),
                    }
                )

    runoff_features_df = pd.DataFrame(
        runoff_feature_rows,
        columns=["runoff_source", "feature_name", "feature_value"],
    )

    metadata_df = pd.DataFrame(
        [
            {
                "key": str(key),
                "value": json.dumps(value) if isinstance(value, (list, dict)) else str(value),
            }
            for key, value in ds.attrs.items()
        ]
    )

    ds.close()

    return {
        "routing_nodes": nodes_df,
        "routing_edges": edges_df,
        "routing_edge_features": edge_features_df,
        "routing_node_features": node_features_df,
        "routing_gauges": routing_gauges_df,
        "runoff_mapping": runoff_mapping_df,
        "runoff_source_features": runoff_features_df,
        "routing_graph_metadata": metadata_df,
    }


def write_spatial_tables(
    db_path: Path,
    basin_df: pd.DataFrame,
    graph_tables: dict[str, pd.DataFrame],
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    con = duckdb.connect(str(db_path))

    table_names = [
        "basin_boundary",
        "routing_nodes",
        "routing_edges",
        "routing_edge_features",
        "routing_node_features",
        "routing_gauges",
        "runoff_mapping",
        "runoff_source_features",
        "routing_graph_metadata",
    ]

    for table_name in table_names:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")

    con.register("basin_df", basin_df)
    con.execute("CREATE TABLE basin_boundary AS SELECT * FROM basin_df")

    for table_name, df in graph_tables.items():
        view_name = f"{table_name}_df"
        con.register(view_name, df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view_name}")

    con.execute(
        """
        CREATE OR REPLACE VIEW gauge_locations_combined AS
        SELECT
            COALESCE(g.gauge_id, rg.gauge_id) AS gauge_id,
            COALESCE(g.lat, rg.lat) AS lat,
            COALESCE(g.lon, rg.lon) AS lon,
            rg.node_id,
            rg.node_y,
            rg.node_x,
            rg.flat_index
        FROM gauges g
        FULL OUTER JOIN routing_gauges rg
            ON g.gauge_id = rg.gauge_id
        """
    )

    print("\nSpatial tables written.")
    print(con.sql("SHOW TABLES").fetchdf())

    print("\nCounts:")
    print(
        con.sql(
            """
            SELECT
                (SELECT COUNT(*) FROM basin_boundary) AS n_basin_features,
                (SELECT COUNT(*) FROM routing_nodes) AS n_nodes,
                (SELECT COUNT(*) FROM routing_edges) AS n_edges,
                (SELECT COUNT(*) FROM routing_edge_features) AS n_edge_features,
                (SELECT COUNT(*) FROM routing_node_features) AS n_node_features,
                (SELECT COUNT(*) FROM routing_gauges) AS n_routing_gauges,
                (SELECT COUNT(*) FROM runoff_mapping) AS n_runoff_mappings,
                (SELECT COUNT(*) FROM runoff_source_features) AS n_runoff_source_features
            """
        ).fetchdf()
    )

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--basin-shp", type=Path, default=DEFAULT_BASIN_SHP)
    parser.add_argument("--graph-nc", type=Path, default=DEFAULT_GRAPH_NC)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    basin_df = build_basin_boundary_table(args.basin_shp)
    graph_tables = build_graph_tables(args.graph_nc, project_root=args.project_root)

    write_spatial_tables(
        db_path=args.db_path,
        basin_df=basin_df,
        graph_tables=graph_tables,
    )


if __name__ == "__main__":
    main()
