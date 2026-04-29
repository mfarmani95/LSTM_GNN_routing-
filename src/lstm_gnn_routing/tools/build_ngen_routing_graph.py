from __future__ import annotations

import argparse
from collections import deque
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lstm_gnn_routing.routing_models.graph_builder import export_routing_graph_netcdf

logger = logging.getLogger("build_ngen_routing_graph")


DEFAULT_NGEN_ROOT = Path("Input/River_netwrok")
DEFAULT_DEM = Path("Input/SRTM/basin_srtm_dem_conditioned_on_forcing_grid.nc")
DEFAULT_BASIN_FILE = Path("Input/Streamflow/30_basin_ids.txt")
DEFAULT_GAUGE_METADATA = Path("Input/Streamflow/30_gauges_IN_LAMBERT.csv")
DEFAULT_OUTPUT = Path("Pytorch/routing_graph_ngen_salt_verde_cache.nc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Ngen flowpath routing graph and grid-to-flowpath runoff-transfer "
            "mapping on the DEM/forcing grid."
        )
    )
    parser.add_argument("--ngen-root", type=Path, default=DEFAULT_NGEN_ROOT)
    parser.add_argument(
        "--outlet-gauges",
        nargs="+",
        default=["09511300", "09510000", "09510200"],
        help="Independent downstream outlet gauge IDs whose Ngen subsets define the full domain.",
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--dem-variable", default="DEM")
    parser.add_argument("--basin-file", type=Path, default=DEFAULT_BASIN_FILE)
    parser.add_argument("--gauge-metadata", type=Path, default=DEFAULT_GAUGE_METADATA)
    parser.add_argument(
        "--exclude-gauges",
        nargs="*",
        default=[],
        help="Gauge IDs to omit from the graph gauge mapping, for example gauges removed from training.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpkg-template",
        default="gage-{gage}/config/gage-{gage}_subset.gpkg",
        help="Path template under --ngen-root for each gauge subset GeoPackage.",
    )
    parser.add_argument(
        "--allow-missing-gauges",
        action="store_true",
        help="Write gauges that can be mapped and warn for missing gauges instead of failing.",
    )
    parser.add_argument(
        "--source-feature",
        action="append",
        default=["elevation", "cell_area_m2", "distance_to_flowpath_m"],
        choices=[
            "elevation",
            "cell_area_m2",
            "overlap_area_m2",
            "overlap_fraction",
            "distance_to_flowpath_m",
            "x",
            "y",
        ],
        help="Runoff source feature to store for optional neural transfer. May be repeated.",
    )
    parser.add_argument(
        "--runoff-mapping-method",
        choices=["center", "fractional_area"],
        default="center",
        help=(
            "How grid runoff sources are mapped to Ngen divides. 'center' assigns each active grid-cell center "
            "to one containing divide. 'fractional_area' intersects grid-cell polygons with divides and stores "
            "one source row per positive grid-cell/divide overlap."
        ),
    )
    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=1.0e-8,
        help="Minimum grid-cell area fraction to keep when --runoff-mapping-method=fractional_area.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _require_geospatial():
    try:
        import geopandas as gpd
        from pyproj import CRS
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "This tool requires geopandas, pyproj, and shapely. Install them in the preprocessing environment."
        ) from exc
    return gpd, CRS, Point


def _read_dem_grid(path: Path, variable: str) -> dict[str, Any]:
    _, CRS, _ = _require_geospatial()
    ds = xr.open_dataset(path)
    try:
        if variable not in ds:
            raise KeyError(f"DEM variable '{variable}' not found in {path}")
        dem = np.asarray(ds[variable].values, dtype=np.float32)
        if dem.ndim != 2:
            raise ValueError(f"DEM variable must be 2-D, got shape {dem.shape}")

        if "x" in ds and "y" in ds:
            x_values = np.asarray(ds["x"].values)
            y_values = np.asarray(ds["y"].values)
            if x_values.ndim == 1 and y_values.ndim == 1:
                x2d, y2d = np.meshgrid(x_values, y_values, indexing="xy")
            else:
                x2d = np.asarray(x_values)
                y2d = np.asarray(y_values)
        else:
            yy, xx = np.meshgrid(np.arange(dem.shape[0]), np.arange(dem.shape[1]), indexing="ij")
            y2d = yy.astype(np.float32)
            x2d = xx.astype(np.float32)

        crs = None
        for coord_name in ("spatial_ref", "crs"):
            if coord_name in ds:
                attrs = dict(ds[coord_name].attrs)
                wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
                if wkt:
                    crs = CRS.from_wkt(str(wkt))
                    break
        if crs is None:
            raise ValueError(
                f"Could not infer DEM CRS from {path}. Expected spatial_ref/crs attrs with WKT."
            )

        active_mask = np.isfinite(dem)
        if not bool(active_mask.any()):
            raise ValueError("DEM contains no finite active cells")

        dx = float(np.nanmedian(np.abs(np.diff(x2d, axis=1)))) if x2d.shape[1] > 1 else 1.0
        dy = float(np.nanmedian(np.abs(np.diff(y2d, axis=0)))) if y2d.shape[0] > 1 else 1.0
        cell_area_m2 = abs(dx * dy)
        return {
            "dem": dem,
            "x2d": x2d.astype(np.float64),
            "y2d": y2d.astype(np.float64),
            "crs": crs,
            "active_mask": active_mask,
            "cell_area_m2": float(cell_area_m2),
        }
    finally:
        ds.close()


def _read_attr_table(gpkg: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(f'select * from "{table}"', con)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _gpkg_path(root: Path, template: str, gauge: str) -> Path:
    gauge_text = str(gauge)
    return root / template.format(gauge=gauge_text, gage=gauge_text)


def _read_ngen_subsets(root: Path, outlet_gauges: Iterable[str], template: str, target_crs: Any) -> dict[str, Any]:
    gpd, _, _ = _require_geospatial()
    frames: dict[str, list[Any]] = {
        "flowpaths": [],
        "divides": [],
        "nexus": [],
        "hydrolocations": [],
        "flowpath_attributes": [],
        "divide_attributes": [],
    }
    for gauge in outlet_gauges:
        gpkg = _gpkg_path(root, template, str(gauge))
        if not gpkg.is_file():
            raise FileNotFoundError(f"Ngen subset GeoPackage not found for outlet {gauge}: {gpkg}")
        logger.info("Reading Ngen subset %s", gpkg)
        for layer in ("flowpaths", "divides", "nexus", "hydrolocations"):
            try:
                gdf = gpd.read_file(gpkg, layer=layer)
            except Exception as exc:
                logger.warning("Could not read layer '%s' from %s: %s", layer, gpkg, exc)
                continue
            if gdf.empty:
                continue
            if gdf.crs is None:
                raise ValueError(f"Layer '{layer}' in {gpkg} has no CRS")
            frames[layer].append(gdf.to_crs(target_crs))
        frames["flowpath_attributes"].append(_read_attr_table(gpkg, "flowpath-attributes"))
        frames["divide_attributes"].append(_read_attr_table(gpkg, "divide-attributes"))

    result: dict[str, Any] = {}
    for layer in ("flowpaths", "divides", "nexus", "hydrolocations"):
        if not frames[layer]:
            result[layer] = gpd.GeoDataFrame()
            continue
        gdf = pd.concat(frames[layer], ignore_index=True)
        key = "divide_id" if layer == "divides" else "id"
        if key in gdf:
            gdf = gdf.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
        result[layer] = gdf

    for key, table_key in (("flowpath_attributes", "id"), ("divide_attributes", "divide_id")):
        tables = [df for df in frames[key] if isinstance(df, pd.DataFrame) and not df.empty]
        if tables:
            df = pd.concat(tables, ignore_index=True)
            if table_key in df:
                df = df.drop_duplicates(subset=[table_key], keep="first").reset_index(drop=True)
        else:
            df = pd.DataFrame()
        result[key] = df
    return result


def _read_basin_ids(path: Path) -> list[str]:
    with path.open("r") as fp:
        return [line.strip() for line in fp if line.strip()]


def _filter_basin_ids(basin_ids: Iterable[str], exclude_gauges: Iterable[str]) -> list[str]:
    excluded = {str(value).strip() for value in exclude_gauges if str(value).strip()}
    return [str(value) for value in basin_ids if str(value) not in excluded]


def _nearest_flat_indices(points_x: np.ndarray, points_y: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = x2d[0]
    y_axis = y2d[:, 0]
    x_idx = np.abs(points_x[:, None] - x_axis[None, :]).argmin(axis=1).astype(np.int64)
    y_idx = np.abs(points_y[:, None] - y_axis[None, :]).argmin(axis=1).astype(np.int64)
    flat = y_idx * int(x2d.shape[1]) + x_idx
    return flat.astype(np.int64), y_idx, x_idx


def _axis_cell_bounds(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.size == 0:
        raise ValueError("Cannot build grid-cell polygons from an empty coordinate axis")
    if axis.size == 1:
        return axis - 0.5, axis + 0.5

    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - (edges[1] - axis[0])
    edges[-1] = axis[-1] + (axis[-1] - edges[-2])
    lower = np.minimum(edges[:-1], edges[1:])
    upper = np.maximum(edges[:-1], edges[1:])
    return lower, upper


def _active_grid_cell_polygons(grid: Mapping[str, Any]):
    gpd, _, _ = _require_geospatial()
    from shapely.geometry import box

    active_y, active_x = np.where(grid["active_mask"])
    source_flat = (active_y * int(grid["dem"].shape[1]) + active_x).astype(np.int64)
    x_lower, x_upper = _axis_cell_bounds(grid["x2d"][0])
    y_lower, y_upper = _axis_cell_bounds(grid["y2d"][:, 0])
    geometry = [
        box(float(x_lower[x]), float(y_lower[y]), float(x_upper[x]), float(y_upper[y]))
        for y, x in zip(active_y, active_x)
    ]
    return gpd.GeoDataFrame(
        {
            "source_flat_index": source_flat,
            "source_y": active_y.astype(np.int64),
            "source_x": active_x.astype(np.int64),
            "source_cell_area_m2": np.asarray([geom.area for geom in geometry], dtype=np.float64),
            "elevation": grid["dem"][active_y, active_x].astype(np.float32),
            "source_center_x": grid["x2d"][active_y, active_x].astype(np.float64),
            "source_center_y": grid["y2d"][active_y, active_x].astype(np.float64),
        },
        geometry=geometry,
        crs=grid["crs"],
    )


def _numeric(values: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _first_available_numeric(
    frame: pd.DataFrame,
    names: Iterable[str],
    *,
    default: float = 0.0,
) -> np.ndarray:
    for name in names:
        if name in frame:
            return _numeric(frame[name], default=default)
    return np.full(len(frame), float(default), dtype=np.float32)


def _topological_order_from_successor(successor: np.ndarray) -> list[int]:
    num_nodes = int(successor.size)
    indegree = np.zeros(num_nodes, dtype=np.int64)
    for target in successor:
        if int(target) >= 0:
            indegree[int(target)] += 1

    queue: deque[int] = deque(int(index) for index in np.where(indegree == 0)[0].tolist())
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        downstream = int(successor[node])
        if downstream >= 0:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    if len(order) != num_nodes:
        remaining = [int(index) for index in range(num_nodes) if index not in set(order)]
        logger.warning(
            "Routing graph topology is not a clean DAG for all nodes; appending %s unresolved nodes to topological order.",
            len(remaining),
        )
        order.extend(remaining)
    return order


def _derive_node_hydrology_features(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    flowpaths: pd.DataFrame,
    local_length_m: np.ndarray,
    local_slope: np.ndarray,
    local_area_km2: np.ndarray,
    total_area_km2: np.ndarray,
    stream_order: np.ndarray,
    musk_values: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    successor = np.full(num_nodes, -1, dtype=np.int64)
    if edge_index.numel():
        edge_np = edge_index.cpu().numpy()
        for source, target in zip(edge_np[0], edge_np[1]):
            source_idx = int(source)
            target_idx = int(target)
            if successor[source_idx] >= 0 and successor[source_idx] != target_idx:
                logger.warning(
                    "Node %s has multiple downstream targets (%s, %s); keeping the first target for derived features.",
                    source_idx,
                    int(successor[source_idx]),
                    target_idx,
                )
                continue
            successor[source_idx] = target_idx

    topo_order = _topological_order_from_successor(successor)
    positive_slope = np.maximum(local_slope.astype(np.float32), 1.0e-6)
    valid_musk = (
        musk_values is not None
        and np.isfinite(musk_values).any()
        and float(np.nanmax(musk_values)) > 0.0
        and float(np.nanstd(musk_values)) > 1.0e-6
    )
    if valid_musk:
        local_travel_time_proxy = np.maximum(np.asarray(musk_values, dtype=np.float32), 1.0e-6)
        travel_time_source = "MusK"
    else:
        local_travel_time_proxy = (
            np.maximum(local_length_m.astype(np.float32), 1.0) / np.sqrt(positive_slope)
        ).astype(np.float32)
        travel_time_source = "length_over_sqrt_slope"

    distance_to_outlet_m = np.maximum(local_length_m.astype(np.float32), 0.0).copy()
    travel_time_proxy_to_outlet = local_travel_time_proxy.copy()
    upstream_flowpath_count = np.ones(num_nodes, dtype=np.float32)
    upstream_area_km2 = np.maximum(local_area_km2.astype(np.float32), 0.0).copy()

    for node in topo_order:
        downstream = int(successor[node])
        if downstream >= 0:
            upstream_flowpath_count[downstream] += upstream_flowpath_count[node]
            upstream_area_km2[downstream] += upstream_area_km2[node]

    for node in reversed(topo_order):
        downstream = int(successor[node])
        if downstream >= 0:
            distance_to_outlet_m[node] = local_length_m[node] + distance_to_outlet_m[downstream]
            travel_time_proxy_to_outlet[node] = (
                local_travel_time_proxy[node] + travel_time_proxy_to_outlet[downstream]
            )

    safe_total_area = np.maximum(total_area_km2.astype(np.float32), 1.0e-6)
    incremental_area_fraction = np.clip(
        np.maximum(local_area_km2.astype(np.float32), 0.0) / safe_total_area,
        0.0,
        1.0,
    )
    upstream_area_ratio = np.clip(
        np.maximum(total_area_km2.astype(np.float32) - local_area_km2.astype(np.float32), 0.0) / safe_total_area,
        0.0,
        1.0,
    )
    component_ids = _component_ids(num_nodes, edge_index)
    normalized_stream_order = np.zeros(num_nodes, dtype=np.float32)
    for component_id in np.unique(component_ids):
        mask = component_ids == int(component_id)
        component_max = float(np.nanmax(stream_order[mask])) if bool(mask.any()) else 0.0
        if component_max > 0.0:
            normalized_stream_order[mask] = stream_order[mask] / component_max
    derived = {
        "local_length_m": local_length_m.astype(np.float32),
        "distance_to_outlet_m": distance_to_outlet_m.astype(np.float32),
        "local_travel_time_proxy": local_travel_time_proxy.astype(np.float32),
        "travel_time_proxy_to_outlet": travel_time_proxy_to_outlet.astype(np.float32),
        "incremental_area_fraction": incremental_area_fraction.astype(np.float32),
        "upstream_area_ratio": upstream_area_ratio.astype(np.float32),
        "upstream_flowpath_count": upstream_flowpath_count.astype(np.float32),
        "upstream_area_km2_topologic": upstream_area_km2.astype(np.float32),
        "normalized_stream_order": normalized_stream_order.astype(np.float32),
        "is_outlet_node": (successor < 0).astype(np.float32),
    }
    metadata_notes = [
        "distance_to_outlet_m",
        "travel_time_proxy_to_outlet",
        "incremental_area_fraction",
        "upstream_area_ratio",
        "upstream_flowpath_count",
        "normalized_stream_order",
        f"travel_time_proxy_source={travel_time_source}",
    ]
    return derived, metadata_notes


def _build_graph_nodes_edges(ngen: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    flowpaths = ngen["flowpaths"].copy()
    nexus = ngen["nexus"].copy()
    flow_attrs = ngen["flowpath_attributes"].copy()
    divide_attrs = ngen["divide_attributes"].copy()
    if flowpaths.empty:
        raise ValueError("No flowpaths were read from the selected Ngen subsets")
    if "id" not in flowpaths or "toid" not in flowpaths:
        raise ValueError("Ngen flowpaths layer must contain 'id' and 'toid'")

    flowpaths["id"] = flowpaths["id"].astype(str)
    flowpaths["toid"] = flowpaths["toid"].astype(str)
    if "divide_id" in flowpaths:
        flowpaths["divide_id"] = flowpaths["divide_id"].astype(str)
    if not flow_attrs.empty and "id" in flow_attrs:
        flow_attrs["id"] = flow_attrs["id"].astype(str)
        flowpaths = flowpaths.merge(flow_attrs, on="id", how="left", suffixes=("", "_attr"))
    if not divide_attrs.empty and "divide_id" in divide_attrs and "divide_id" in flowpaths:
        divide_attrs["divide_id"] = divide_attrs["divide_id"].astype(str)
        selected_cols = [
            col
            for col in ("divide_id", "mean.elevation", "mean.slope", "mean.impervious", "mode.ISLTYP", "mode.IVGTYP")
            if col in divide_attrs
        ]
        if len(selected_cols) > 1:
            flowpaths = flowpaths.merge(divide_attrs[selected_cols], on="divide_id", how="left")

    node_ids = flowpaths["id"].tolist()
    node_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

    reps = flowpaths.geometry.representative_point()
    rep_x = np.asarray([geom.x for geom in reps], dtype=np.float64)
    rep_y = np.asarray([geom.y for geom in reps], dtype=np.float64)
    flat_index, node_y, node_x = _nearest_flat_indices(rep_x, rep_y, grid["x2d"], grid["y2d"])
    node_dem_elevation = grid["dem"][node_y, node_x].astype(np.float32)

    nexus_to_downstream = {}
    if not nexus.empty and "id" in nexus and "toid" in nexus:
        nexus_to_downstream = {
            str(row.id): str(row.toid)
            for row in nexus[["id", "toid"]].itertuples(index=False)
            if pd.notna(row.id) and pd.notna(row.toid)
        }

    edge_pairs: list[tuple[int, int]] = []
    edge_source_ids: list[str] = []
    for row in flowpaths[["id", "toid"]].itertuples(index=False):
        source_id = str(row.id)
        downstream_id = nexus_to_downstream.get(str(row.toid))
        if downstream_id is None or downstream_id not in node_map:
            continue
        source = node_map[source_id]
        target = node_map[downstream_id]
        if source == target:
            continue
        edge_pairs.append((source, target))
        edge_source_ids.append(source_id)

    if not edge_pairs:
        raise ValueError("No directed flowpath edges were built. Check Ngen nexus topology.")
    edge_df = pd.DataFrame(edge_pairs, columns=["source", "target"])
    edge_df["source_id"] = edge_source_ids
    edge_df = edge_df.drop_duplicates(subset=["source", "target"], keep="first").reset_index(drop=True)
    edge_index = torch.as_tensor(edge_df[["source", "target"]].to_numpy().T, dtype=torch.long)

    attr_by_id = flowpaths.set_index("id")
    edge_rows = attr_by_id.loc[edge_df["source_id"].tolist()]
    length_m = (
        _numeric(edge_rows["Length_m"])
        if "Length_m" in edge_rows
        else _numeric(edge_rows.get("lengthkm", pd.Series(index=edge_rows.index)), default=0.0) * 1000.0
    )
    slope = (
        _numeric(edge_rows["So"], default=0.0)
        if "So" in edge_rows
        else _numeric(edge_rows.get("ChSlp", edge_rows.get("mean.slope", pd.Series(index=edge_rows.index))), default=0.0)
    )
    edge_feature_arrays: list[np.ndarray] = [length_m, slope]
    edge_feature_names: list[str] = ["Length_m", "So"]

    optional_edge_columns = [
        ("n", "n"),
        ("nCC", "nCC"),
        ("BtmWdth", "BtmWdth"),
        ("TopWdth", "TopWdth"),
        ("TopWdthCC", "TopWdthCC"),
        ("MusX", "MusX"),
        ("MusK", "MusK"),
    ]
    optional_edge_values: dict[str, np.ndarray] = {}
    for column_name, feature_name in optional_edge_columns:
        if column_name not in edge_rows:
            continue
        values = _numeric(edge_rows[column_name], default=0.0)
        optional_edge_values[feature_name] = values
        edge_feature_arrays.append(values)
        edge_feature_names.append(feature_name)

    positive_edge_slope = np.maximum(np.abs(slope), 1.0e-6)
    edge_travel_time_proxy = (
        np.maximum(optional_edge_values["MusK"], 1.0e-6)
        if "MusK" in optional_edge_values
        and np.isfinite(optional_edge_values["MusK"]).any()
        and float(np.nanmax(optional_edge_values["MusK"])) > 0.0
        and float(np.nanstd(optional_edge_values["MusK"])) > 1.0e-6
        else np.maximum(length_m, 1.0) / np.sqrt(positive_edge_slope)
    ).astype(np.float32)
    edge_feature_arrays.append(edge_travel_time_proxy)
    edge_feature_names.append("travel_time_proxy")

    roughness_n = optional_edge_values.get("n")
    if roughness_n is None:
        roughness_n = np.full_like(length_m, 0.05, dtype=np.float32)
    roughness_n = np.maximum(np.asarray(roughness_n, dtype=np.float32), 1.0e-6)

    top_width = optional_edge_values.get("TopWdth")
    if top_width is None:
        top_width = optional_edge_values.get("BtmWdth")
    if top_width is None:
        top_width = np.ones_like(length_m, dtype=np.float32)
    top_width = np.maximum(np.asarray(top_width, dtype=np.float32), 1.0e-6)

    bottom_width = optional_edge_values.get("BtmWdth")
    if bottom_width is None:
        bottom_width = top_width
    bottom_width = np.maximum(np.asarray(bottom_width, dtype=np.float32), 1.0e-6)

    sqrt_slope = np.sqrt(positive_edge_slope).astype(np.float32)
    velocity_proxy = (sqrt_slope / roughness_n).astype(np.float32)
    manning_travel_time_proxy = (np.maximum(length_m, 1.0) / np.maximum(velocity_proxy, 1.0e-6)).astype(np.float32)
    inverse_manning_travel_time_proxy = (1.0 / np.maximum(manning_travel_time_proxy, 1.0e-6)).astype(np.float32)
    storage_proxy = (np.maximum(length_m, 1.0) * top_width).astype(np.float32)
    conveyance_proxy = (top_width * sqrt_slope / roughness_n).astype(np.float32)
    width_ratio_proxy = (top_width / bottom_width).astype(np.float32)

    for feature_name, values in [
        ("velocity_proxy", velocity_proxy),
        ("manning_travel_time_proxy", manning_travel_time_proxy),
        ("inverse_manning_travel_time_proxy", inverse_manning_travel_time_proxy),
        ("storage_proxy", storage_proxy),
        ("conveyance_proxy", conveyance_proxy),
        ("width_ratio_proxy", width_ratio_proxy),
    ]:
        edge_feature_arrays.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        edge_feature_names.append(feature_name)

    edge_attr = torch.as_tensor(np.stack(edge_feature_arrays, axis=1), dtype=torch.float32)
    musk = optional_edge_values.get("MusK")
    use_variable_musk = bool(
        musk is not None
        and np.isfinite(musk).any()
        and float(np.nanmax(musk)) > 0.0
        and float(np.nanstd(musk)) > 1.0e-6
    )
    edge_weight_base = (
        1.0 / np.maximum(musk, 1.0e-6)
        if use_variable_musk
        else 1.0 / np.maximum(length_m, 1.0)
    )
    edge_weight = torch.as_tensor(edge_weight_base.astype(np.float32), dtype=torch.float32)

    local_length_m = _first_available_numeric(flowpaths, ("Length_m",), default=np.nan)
    if not np.isfinite(local_length_m).any() or float(np.nanmax(local_length_m)) <= 0.0:
        local_length_m = _first_available_numeric(flowpaths, ("lengthkm",), default=0.0) * 1000.0
    else:
        local_length_m = np.nan_to_num(local_length_m, nan=0.0, posinf=0.0, neginf=0.0)
    local_slope = _first_available_numeric(flowpaths, ("So", "ChSlp", "mean.slope"), default=0.0)
    local_area_km2 = _first_available_numeric(flowpaths, ("areasqkm",), default=0.0)
    total_area_km2 = _first_available_numeric(flowpaths, ("tot_drainage_areasqkm", "TotDASqKM"), default=0.0)
    stream_order = _first_available_numeric(flowpaths, ("order", "streamorde"), default=0.0)
    musk_by_node = _first_available_numeric(flowpaths, ("MusK",), default=0.0) if "MusK" in flowpaths else None
    derived_node_features, derived_node_feature_notes = _derive_node_hydrology_features(
        num_nodes=len(node_ids),
        edge_index=edge_index,
        flowpaths=flowpaths,
        local_length_m=local_length_m,
        local_slope=local_slope,
        local_area_km2=local_area_km2,
        total_area_km2=total_area_km2,
        stream_order=stream_order,
        musk_values=musk_by_node,
    )

    node_features = [node_dem_elevation]
    node_feature_names = ["node_dem_elevation_m"]
    node_feature_columns = [
        "lengthkm",
        "areasqkm",
        "tot_drainage_areasqkm",
        "order",
        "mean.elevation",
        "mean.slope",
        "mean.impervious",
        "mode.ISLTYP",
        "mode.IVGTYP",
    ]
    for col in node_feature_columns:
        if col in flowpaths:
            node_features.append(_numeric(flowpaths[col]))
            node_feature_names.append(col)
    for name, values in derived_node_features.items():
        node_features.append(values)
        node_feature_names.append(name)
    if node_features:
        node_features_np = np.stack(node_features, axis=1).astype(np.float32)
        node_features_np = np.nan_to_num(node_features_np, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        node_features_np = np.zeros((len(node_ids), 1), dtype=np.float32)
        node_feature_names = ["constant"]

    payload = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_feature_names": edge_feature_names,
        "edge_weight": edge_weight,
        "flat_index": torch.as_tensor(flat_index, dtype=torch.long),
        "node_y": torch.as_tensor(node_y, dtype=torch.long),
        "node_x": torch.as_tensor(node_x, dtype=torch.long),
        "node_ids": node_ids,
        "node_features": torch.as_tensor(node_features_np, dtype=torch.float32),
        "node_feature_names": node_feature_names,
        "metadata": {
            "builder": "ngen_flowpath",
            "node_feature_names": node_feature_names,
            "derived_node_features": derived_node_feature_notes,
            "edge_weight_feature": "inverse_MusK_if_variable_else_length",
            "grid_shape": tuple(int(v) for v in grid["dem"].shape),
        },
    }
    return payload


def _map_gauges(
    payload: dict[str, Any],
    ngen: Mapping[str, Any],
    basin_ids: list[str],
    gauge_metadata: Path,
    *,
    allow_missing: bool,
) -> None:
    gpd, _, Point = _require_geospatial()
    node_map = {str(node_id): idx for idx, node_id in enumerate(payload["node_ids"])}
    gauge_to_wb: dict[str, str] = {}
    flowpaths = ngen["flowpaths"]
    divides = ngen["divides"]

    hydrolocations = ngen["hydrolocations"]
    if not hydrolocations.empty:
        for row in hydrolocations.itertuples(index=False):
            row_dict = row._asdict()
            ref = str(row_dict.get("hl_reference", ""))
            uri = str(row_dict.get("hl_uri", ""))
            gauge_id = None
            if ref == "gages" and uri.startswith("gages-"):
                gauge_id = uri.replace("gages-", "")
            elif str(row_dict.get("hl_link", "")).isdigit():
                gauge_id = str(row_dict.get("hl_link"))
            wb_id = row_dict.get("id")
            if gauge_id and wb_id:
                gauge_to_wb[gauge_id] = str(wb_id)

    flow_attrs = ngen["flowpath_attributes"]
    if not flow_attrs.empty and "gage" in flow_attrs and "id" in flow_attrs:
        for row in flow_attrs[["gage", "id"]].dropna().itertuples(index=False):
            gauge_to_wb[str(row.gage)] = str(row.id)

    metadata = pd.DataFrame()
    if gauge_metadata.is_file():
        metadata = pd.read_csv(gauge_metadata, dtype={"basin_id": str})
        if "basin_id" in metadata:
            metadata["basin_id"] = metadata["basin_id"].astype(str)
            metadata = metadata[metadata["basin_id"].isin([str(value) for value in basin_ids])].copy()

    if not metadata.empty and {"x", "y"}.issubset(metadata.columns) and not divides.empty and not flowpaths.empty:
        divide_to_wb: dict[str, str] = {}
        if "divide_id" in divides and "divide_id" in flowpaths:
            flowpaths_for_divides = flowpaths[["id", "divide_id"]].copy()
            flowpaths_for_divides["id"] = flowpaths_for_divides["id"].astype(str)
            flowpaths_for_divides["divide_id"] = flowpaths_for_divides["divide_id"].astype(str)
            divide_to_wb = dict(zip(flowpaths_for_divides["divide_id"], flowpaths_for_divides["id"]))
        if divide_to_wb:
            points = [Point(float(row.x), float(row.y)) for row in metadata.itertuples(index=False)]
            metadata_points = gpd.GeoDataFrame(metadata.copy(), geometry=points, crs=flowpaths.crs)
            gauge_divides = gpd.sjoin(
                metadata_points,
                divides[["divide_id", "geometry"]],
                how="left",
                predicate="within",
            )
            for row in gauge_divides.dropna(subset=["divide_id"]).itertuples(index=False):
                gauge_id = str(getattr(row, "basin_id"))
                wb_id = divide_to_wb.get(str(getattr(row, "divide_id")))
                if wb_id and wb_id not in gauge_to_wb:
                    gauge_to_wb[gauge_id] = wb_id
                    logger.info(
                        "Gauge %s mapped through containing Ngen divide %s",
                        gauge_id,
                        getattr(row, "divide_id"),
                    )
            missing_divide_gauges = [
                str(value)
                for value in metadata["basin_id"].tolist()
                if str(value) not in gauge_to_wb
            ]
            if missing_divide_gauges:
                divides_by_id = divides.set_index(divides["divide_id"].astype(str))
                for gauge_id in missing_divide_gauges:
                    row = metadata.loc[metadata["basin_id"] == gauge_id]
                    if row.empty:
                        continue
                    point = Point(float(row.iloc[0]["x"]), float(row.iloc[0]["y"]))
                    distances = divides_by_id.geometry.distance(point)
                    if distances.empty:
                        continue
                    nearest_divide = str(distances.idxmin())
                    wb_id = divide_to_wb.get(nearest_divide)
                    if wb_id and wb_id not in gauge_to_wb:
                        gauge_to_wb[gauge_id] = wb_id
                        logger.warning(
                            "Gauge %s mapped through nearest Ngen divide %s, not by containing divide.",
                            gauge_id,
                            nearest_divide,
                        )

    gauge_indices: list[int] = []
    gauge_ids: list[str] = []
    missing: list[str] = []
    for basin_id in basin_ids:
        wb_id = gauge_to_wb.get(str(basin_id))
        if wb_id in node_map:
            gauge_indices.append(node_map[wb_id])
            gauge_ids.append(str(basin_id))
            continue

        if not metadata.empty and {"x", "y"}.issubset(metadata.columns) and not flowpaths.empty:
            row = metadata.loc[metadata["basin_id"] == str(basin_id)]
            if not row.empty:
                x_val = float(row.iloc[0]["x"])
                y_val = float(row.iloc[0]["y"])
                point = Point(x_val, y_val)
                distances = flowpaths.geometry.distance(point)
                nearest_pos = int(distances.idxmin())
                nearest_wb = str(flowpaths.loc[nearest_pos, "id"])
                if nearest_wb in node_map:
                    logger.warning(
                        "Gauge %s was mapped by nearest projected flowpath, not by Ngen hydrolocation.",
                        basin_id,
                    )
                    gauge_indices.append(node_map[nearest_wb])
                    gauge_ids.append(str(basin_id))
                    continue

        missing.append(str(basin_id))

    if missing and not allow_missing:
        raise ValueError(
            "Could not map all target basin ids to Ngen flowpath nodes. "
            f"Missing: {missing}. Check hydrolocations/flowpath-attributes or provide a gauge crosswalk."
        )
    if missing:
        logger.warning("Skipping %s gauges that could not be mapped: %s", len(missing), missing)

    duplicates = pd.Series(gauge_indices).value_counts()
    duplicate_nodes = duplicates[duplicates > 1]
    if not duplicate_nodes.empty:
        logger.warning(
            "Multiple gauges mapped to the same graph node. This is okay only if the graph resolution "
            "cannot distinguish them; consider splitting flowpaths for internal gauges. node_counts=%s",
            duplicate_nodes.to_dict(),
        )

    payload["gauge_index"] = torch.as_tensor(gauge_indices, dtype=torch.long)
    payload["gauge_ids"] = gauge_ids


def _center_grid_sources_to_divides(divides, grid: Mapping[str, Any]):
    gpd, _, Point = _require_geospatial()
    active_y, active_x = np.where(grid["active_mask"])
    source_flat = (active_y * int(grid["dem"].shape[1]) + active_x).astype(np.int64)
    points = [Point(float(grid["x2d"][y, x]), float(grid["y2d"][y, x])) for y, x in zip(active_y, active_x)]
    source_gdf = gpd.GeoDataFrame(
        {
            "source_flat_index": source_flat,
            "source_y": active_y.astype(np.int64),
            "source_x": active_x.astype(np.int64),
            "source_cell_area_m2": np.full(len(source_flat), float(grid["cell_area_m2"]), dtype=np.float64),
            "elevation": grid["dem"][active_y, active_x].astype(np.float32),
            "source_center_x": grid["x2d"][active_y, active_x].astype(np.float64),
            "source_center_y": grid["y2d"][active_y, active_x].astype(np.float64),
        },
        geometry=points,
        crs=grid["crs"],
    )
    join_cols = ["divide_id", "geometry"]
    if "id" in divides:
        join_cols.append("id")
    logger.info("Spatially joining %s active grid cells to Ngen divides", len(source_gdf))
    joined = gpd.sjoin(source_gdf, divides[join_cols], how="inner", predicate="within")
    if joined.empty:
        logger.warning("No grid-cell centers fell within divides using within; retrying intersects")
        joined = gpd.sjoin(source_gdf, divides[join_cols], how="inner", predicate="intersects")
    if joined.empty:
        raise ValueError("No active grid cells were mapped to Ngen divides")
    joined = joined.drop_duplicates(subset=["source_flat_index"], keep="first").copy()
    joined["overlap_area_m2"] = joined["source_cell_area_m2"].astype(np.float64)
    joined["overlap_fraction"] = 1.0
    return joined


def _fractional_area_grid_sources_to_divides(
    divides,
    grid: Mapping[str, Any],
    *,
    min_overlap_fraction: float,
):
    gpd, _, _ = _require_geospatial()
    source_gdf = _active_grid_cell_polygons(grid)
    join_cols = ["divide_id", "geometry"]
    if "id" in divides:
        join_cols.append("id")

    logger.info(
        "Overlaying %s active grid-cell polygons with %s Ngen divides for fractional-area runoff transfer",
        len(source_gdf),
        len(divides),
    )
    joined = gpd.overlay(source_gdf, divides[join_cols], how="intersection", keep_geom_type=False)
    if joined.empty:
        raise ValueError("No active grid-cell polygons intersected Ngen divides")

    joined["overlap_area_m2"] = joined.geometry.area.astype(np.float64)
    joined["overlap_fraction"] = joined["overlap_area_m2"] / np.maximum(
        joined["source_cell_area_m2"].astype(np.float64),
        np.finfo(np.float64).eps,
    )
    min_fraction = max(0.0, float(min_overlap_fraction))
    joined = joined[
        (joined["overlap_area_m2"] > 0.0)
        & np.isfinite(joined["overlap_fraction"])
        & (joined["overlap_fraction"] >= min_fraction)
    ].copy()
    if joined.empty:
        raise ValueError(
            "All grid-cell/divide intersections were removed by min_overlap_fraction="
            f"{min_fraction}"
        )
    return joined


def _map_grid_sources_to_flowpaths(
    payload: dict[str, Any],
    ngen: Mapping[str, Any],
    grid: Mapping[str, Any],
    source_features: list[str],
    *,
    mapping_method: str = "center",
    min_overlap_fraction: float = 1.0e-8,
) -> None:
    _, _, Point = _require_geospatial()
    divides = ngen["divides"].copy()
    flowpaths = ngen["flowpaths"].copy()
    if divides.empty:
        raise ValueError("Cannot build runoff transfer mapping without Ngen divides")
    if "divide_id" not in divides:
        raise ValueError("Ngen divides layer must contain divide_id")
    if "divide_id" not in flowpaths:
        raise ValueError("Ngen flowpaths layer must contain divide_id")

    flowpaths["id"] = flowpaths["id"].astype(str)
    flowpaths["divide_id"] = flowpaths["divide_id"].astype(str)
    divides["divide_id"] = divides["divide_id"].astype(str)
    divide_to_wb = dict(zip(flowpaths["divide_id"], flowpaths["id"]))
    node_map = {str(node_id): idx for idx, node_id in enumerate(payload["node_ids"])}

    method = str(mapping_method or "center").lower()
    if method == "center":
        joined = _center_grid_sources_to_divides(divides, grid)
        mapping_label = "containing_divide_to_flowpath"
        weight_kind = "cell_area_m2"
    elif method == "fractional_area":
        joined = _fractional_area_grid_sources_to_divides(
            divides,
            grid,
            min_overlap_fraction=min_overlap_fraction,
        )
        mapping_label = "fractional_grid_cell_area_to_divide_to_flowpath"
        weight_kind = "overlap_area_m2"
    else:
        raise ValueError("runoff_mapping_method must be one of: center, fractional_area")

    joined["wb_id"] = joined["divide_id"].map(divide_to_wb)
    joined = joined[joined["wb_id"].isin(node_map)]
    if joined.empty:
        raise ValueError("Grid cells joined to divides, but no divide_id mapped to graph flowpath nodes")
    joined["target_index"] = joined["wb_id"].map(node_map).astype(np.int64)
    joined = joined.sort_values(["source_flat_index", "target_index"]).reset_index(drop=True)

    target_index = joined["target_index"].to_numpy(dtype=np.int64)
    source_flat_index = joined["source_flat_index"].to_numpy(dtype=np.int64)
    source_weight = joined["overlap_area_m2"].to_numpy(dtype=np.float32)
    source_fraction = joined["overlap_fraction"].to_numpy(dtype=np.float32)
    payload["runoff_source_index"] = torch.as_tensor(source_flat_index, dtype=torch.long)
    payload["runoff_source_flat_index"] = torch.as_tensor(source_flat_index, dtype=torch.long)
    payload["runoff_target_index"] = torch.as_tensor(target_index, dtype=torch.long)
    payload["runoff_source_weight"] = torch.as_tensor(source_weight, dtype=torch.float32)
    payload["runoff_source_fraction"] = torch.as_tensor(source_fraction, dtype=torch.float32)

    features = []
    names = []
    if "elevation" in source_features:
        features.append(joined["elevation"].to_numpy(dtype=np.float32))
        names.append("elevation")
    if "cell_area_m2" in source_features:
        features.append(joined["source_cell_area_m2"].to_numpy(dtype=np.float32))
        names.append("cell_area_m2")
    if "overlap_area_m2" in source_features:
        features.append(joined["overlap_area_m2"].to_numpy(dtype=np.float32))
        names.append("overlap_area_m2")
    if "overlap_fraction" in source_features:
        features.append(joined["overlap_fraction"].to_numpy(dtype=np.float32))
        names.append("overlap_fraction")
    if "x" in source_features:
        features.append(joined["source_center_x"].to_numpy(dtype=np.float32))
        names.append("x")
    if "y" in source_features:
        features.append(joined["source_center_y"].to_numpy(dtype=np.float32))
        names.append("y")
    if "distance_to_flowpath_m" in source_features:
        flow_geom = flowpaths.set_index("id").geometry.to_dict()
        distances = []
        for row in joined.itertuples(index=False):
            source_point = Point(float(row.source_center_x), float(row.source_center_y))
            wb_id = str(row.wb_id)
            target_geom = flow_geom.get(wb_id)
            distances.append(float(source_point.distance(target_geom)) if target_geom is not None else 0.0)
        features.append(np.asarray(distances, dtype=np.float32))
        names.append("distance_to_flowpath_m")

    if features:
        payload["runoff_source_features"] = torch.as_tensor(np.stack(features, axis=1), dtype=torch.float32)
        payload["runoff_source_feature_names"] = names

    payload["metadata"] = dict(
        payload.get("metadata", {}),
        runoff_source_count=int(len(joined)),
        runoff_source_unique_grid_cells=int(pd.Series(source_flat_index).nunique()),
        runoff_source_mapping=mapping_label,
        runoff_source_weight_kind=weight_kind,
        runoff_source_fraction_kind="overlap_area_over_grid_cell_area",
        runoff_mapping_method=method,
        min_overlap_fraction=float(min_overlap_fraction),
    )
    fraction_sum = joined.groupby("source_flat_index")["overlap_fraction"].sum()
    logger.info(
        "Mapped %s runoff source rows from %s unique grid cells to %s graph nodes | method=%s | fraction_sum[min=%.4f, mean=%.4f, max=%.4f]",
        len(joined),
        int(fraction_sum.size),
        int(np.unique(target_index).size),
        method,
        float(fraction_sum.min()),
        float(fraction_sum.mean()),
        float(fraction_sum.max()),
    )


def _component_ids(num_nodes: int, edge_index: torch.Tensor) -> np.ndarray:
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)
    comp = np.full(num_nodes, -1, dtype=np.int32)
    current = 0
    for start in range(num_nodes):
        if comp[start] >= 0:
            continue
        stack = [start]
        comp[start] = current
        while stack:
            node = stack.pop()
            for nbr in adjacency[node]:
                if comp[nbr] < 0:
                    comp[nbr] = current
                    stack.append(nbr)
        current += 1
    return comp


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s:%(name)s:%(message)s")

    grid = _read_dem_grid(args.dem, args.dem_variable)
    ngen = _read_ngen_subsets(args.ngen_root, args.outlet_gauges, args.gpkg_template, grid["crs"])
    payload = _build_graph_nodes_edges(ngen, grid)
    basin_ids = _filter_basin_ids(_read_basin_ids(args.basin_file), args.exclude_gauges)
    if args.exclude_gauges:
        logger.info("Excluded gauges from graph mapping: %s", ",".join(str(value) for value in args.exclude_gauges))
    _map_gauges(payload, ngen, basin_ids, args.gauge_metadata, allow_missing=bool(args.allow_missing_gauges))
    _map_grid_sources_to_flowpaths(
        payload,
        ngen,
        grid,
        list(dict.fromkeys(args.source_feature)),
        mapping_method=str(args.runoff_mapping_method),
        min_overlap_fraction=float(args.min_overlap_fraction),
    )

    comp = _component_ids(len(payload["node_ids"]), payload["edge_index"])
    metadata = dict(payload.get("metadata", {}))
    metadata["component_count"] = int(comp.max() + 1) if comp.size else 0
    metadata["outlet_gauges"] = [str(value) for value in args.outlet_gauges]
    metadata["basin_file"] = str(args.basin_file)
    metadata["dem"] = str(args.dem)
    payload["metadata"] = metadata

    logger.info(
        "Built Ngen routing graph | nodes=%s edges=%s gauges=%s components=%s",
        len(payload["node_ids"]),
        int(payload["edge_index"].shape[1]),
        int(torch.as_tensor(payload["gauge_index"]).numel()),
        metadata["component_count"],
    )
    out = export_routing_graph_netcdf(payload, args.output)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
