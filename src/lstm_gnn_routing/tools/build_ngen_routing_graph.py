from __future__ import annotations

import argparse
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
        choices=["elevation", "cell_area_m2", "distance_to_flowpath_m", "x", "y"],
        help="Runoff source feature to store for optional neural transfer. May be repeated.",
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


def _numeric(values: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=np.float32)


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
        else _numeric(edge_rows.get("mean.slope", pd.Series(index=edge_rows.index)), default=0.0)
    )
    musk = _numeric(edge_rows["MusK"], default=0.0) if "MusK" in edge_rows else np.zeros_like(length_m)
    edge_attr = torch.as_tensor(np.stack([length_m, slope, musk], axis=1), dtype=torch.float32)
    edge_weight_base = np.where(musk > 0.0, 1.0 / np.maximum(musk, 1.0e-6), 1.0 / np.maximum(length_m, 1.0))
    edge_weight = torch.as_tensor(edge_weight_base.astype(np.float32), dtype=torch.float32)

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
    node_features = []
    node_feature_names = []
    for col in node_feature_columns:
        if col in flowpaths:
            node_features.append(_numeric(flowpaths[col]))
            node_feature_names.append(col)
    if node_features:
        node_features_np = np.stack(node_features, axis=1).astype(np.float32)
        node_features_np = np.nan_to_num(node_features_np, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        node_features_np = np.zeros((len(node_ids), 1), dtype=np.float32)
        node_feature_names = ["constant"]

    payload = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_feature_names": ["Length_m", "So", "MusK"],
        "edge_weight": edge_weight,
        "flat_index": torch.as_tensor(flat_index, dtype=torch.long),
        "node_y": torch.as_tensor(node_y, dtype=torch.long),
        "node_x": torch.as_tensor(node_x, dtype=torch.long),
        "node_ids": node_ids,
        "node_features": torch.as_tensor(node_features_np, dtype=torch.float32),
        "metadata": {
            "builder": "ngen_flowpath",
            "node_feature_names": node_feature_names,
            "edge_weight_feature": "inverse_MusK_or_length",
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


def _map_grid_sources_to_flowpaths(payload: dict[str, Any], ngen: Mapping[str, Any], grid: Mapping[str, Any], source_features: list[str]) -> None:
    gpd, _, Point = _require_geospatial()
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

    active_y, active_x = np.where(grid["active_mask"])
    source_flat = (active_y * int(grid["dem"].shape[1]) + active_x).astype(np.int64)
    points = [Point(float(grid["x2d"][y, x]), float(grid["y2d"][y, x])) for y, x in zip(active_y, active_x)]
    source_gdf = gpd.GeoDataFrame(
        {
            "source_flat_index": source_flat,
            "source_y": active_y.astype(np.int64),
            "source_x": active_x.astype(np.int64),
            "elevation": grid["dem"][active_y, active_x].astype(np.float32),
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
    joined["wb_id"] = joined["divide_id"].map(divide_to_wb)
    joined = joined[joined["wb_id"].isin(node_map)]
    if joined.empty:
        raise ValueError("Grid cells joined to divides, but no divide_id mapped to graph flowpath nodes")
    joined["target_index"] = joined["wb_id"].map(node_map).astype(np.int64)
    joined = joined.sort_values("source_flat_index").reset_index(drop=True)

    target_index = joined["target_index"].to_numpy(dtype=np.int64)
    source_flat_index = joined["source_flat_index"].to_numpy(dtype=np.int64)
    payload["runoff_source_index"] = torch.arange(len(joined), dtype=torch.long)
    payload["runoff_source_flat_index"] = torch.as_tensor(source_flat_index, dtype=torch.long)
    payload["runoff_target_index"] = torch.as_tensor(target_index, dtype=torch.long)
    payload["runoff_source_weight"] = torch.full((len(joined),), float(grid["cell_area_m2"]), dtype=torch.float32)

    features = []
    names = []
    if "elevation" in source_features:
        features.append(joined["elevation"].to_numpy(dtype=np.float32))
        names.append("elevation")
    if "cell_area_m2" in source_features:
        features.append(np.full(len(joined), float(grid["cell_area_m2"]), dtype=np.float32))
        names.append("cell_area_m2")
    if "x" in source_features:
        features.append(grid["x2d"].reshape(-1)[source_flat_index].astype(np.float32))
        names.append("x")
    if "y" in source_features:
        features.append(grid["y2d"].reshape(-1)[source_flat_index].astype(np.float32))
        names.append("y")
    if "distance_to_flowpath_m" in source_features:
        flow_geom = flowpaths.set_index("id").geometry.to_dict()
        distances = []
        for geom, wb_id in zip(joined.geometry, joined["wb_id"]):
            target_geom = flow_geom.get(wb_id)
            distances.append(float(geom.distance(target_geom)) if target_geom is not None else 0.0)
        features.append(np.asarray(distances, dtype=np.float32))
        names.append("distance_to_flowpath_m")

    if features:
        payload["runoff_source_features"] = torch.as_tensor(np.stack(features, axis=1), dtype=torch.float32)
        payload["runoff_source_feature_names"] = names

    payload["metadata"] = dict(payload.get("metadata", {}), runoff_source_count=int(len(joined)))
    logger.info("Mapped %s grid sources to %s graph nodes", len(joined), int(np.unique(target_index).size))


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
    _map_grid_sources_to_flowpaths(payload, ngen, grid, list(dict.fromkeys(args.source_feature)))

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
