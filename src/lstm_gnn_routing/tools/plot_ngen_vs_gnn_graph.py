from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from pyproj import CRS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the source Ngen river network next to the generated GNN routing graph."
    )
    parser.add_argument("--network", nargs="+", type=Path, required=True, help="Ngen GeoPackage/shapefile paths or directories.")
    parser.add_argument("--network-pattern", default="**/*_subset.gpkg", help="Glob used when a network input is a directory.")
    parser.add_argument("--flowpath-layer", default="flowpaths", help="GeoPackage layer containing Ngen flowpaths.")
    parser.add_argument("--outlet-gauges", nargs="*", default=None, help="Optional gauge IDs used to select gage-* network files.")
    parser.add_argument("--graph", type=Path, required=True, help="Generated routing graph NetCDF cache.")
    parser.add_argument("--dem", type=Path, required=True, help="DEM/grid NetCDF used to map graph node_y/node_x to x/y.")
    parser.add_argument("--dem-variable", default="DEM")
    parser.add_argument("--gauge-metadata", type=Path, default=None, help="Optional CSV with basin_id,x,y columns.")
    parser.add_argument("--output", type=Path, default=Path("docs/figures/ngen_network_vs_gnn_graph.png"))
    parser.add_argument("--no-reproject-network", action="store_true", help="Do not reproject Ngen flowpaths to the DEM/grid CRS.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--edge-alpha", type=float, default=0.45)
    parser.add_argument("--edge-width", type=float, default=0.35)
    parser.add_argument("--network-width", type=float, default=0.45)
    return parser.parse_args()


def _discover_network_files(paths: Iterable[Path], pattern: str, outlet_gauges: list[str] | None) -> list[Path]:
    discovered: list[Path] = []
    gauge_tokens = None if not outlet_gauges else {f"gage-{str(gauge)}" for gauge in outlet_gauges}
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(item for item in path.glob(pattern) if item.is_file())
        else:
            raise FileNotFoundError(f"Ngen network input does not exist: {path}")
        if gauge_tokens:
            candidates = [item for item in candidates if any(token in str(item) for token in gauge_tokens)]
        discovered.extend(candidates)
    unique = {str(path.resolve()): path for path in discovered}
    if not unique:
        raise RuntimeError("No Ngen network files were discovered. Check --network, --network-pattern, and --outlet-gauges.")
    return [unique[key] for key in sorted(unique)]


def _read_flowpaths(files: list[Path], layer: str) -> gpd.GeoDataFrame:
    frames = []
    for path in files:
        if path.suffix.lower() == ".gpkg":
            frame = gpd.read_file(path, layer=layer)
        else:
            frame = gpd.read_file(path)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        frame = frame[frame.geometry.geom_type.isin({"LineString", "MultiLineString"})]
        if not frame.empty:
            frames.append(frame[["geometry"]].copy())
    if not frames:
        raise RuntimeError("No line geometries were found in the selected Ngen network files.")
    network = pd.concat(frames, ignore_index=True)
    network["__wkb"] = network.geometry.to_wkb()
    network = network.drop_duplicates("__wkb").drop(columns="__wkb")
    return gpd.GeoDataFrame(network, geometry="geometry", crs=frames[0].crs)


def _load_dem(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, CRS | None]:
    ds = xr.open_dataset(path)
    try:
        dem = np.asarray(ds[variable].values, dtype=np.float32)
        x_values = np.asarray(ds["x"].values)
        y_values = np.asarray(ds["y"].values)
        if x_values.ndim == 1 and y_values.ndim == 1:
            x2d, y2d = np.meshgrid(x_values, y_values, indexing="xy")
        else:
            x2d, y2d = x_values, y_values
        crs = None
        if "spatial_ref" in ds:
            spatial_attrs = dict(ds["spatial_ref"].attrs)
            wkt = spatial_attrs.get("crs_wkt") or spatial_attrs.get("spatial_ref")
            if wkt:
                crs = CRS.from_wkt(str(wkt))
        return dem, x2d, y2d, crs
    finally:
        ds.close()


def _load_graph(path: Path) -> dict[str, np.ndarray | dict]:
    ds = xr.open_dataset(path)
    try:
        graph = {
            "edge_source": np.asarray(ds["edge_source"].values, dtype=np.int64),
            "edge_target": np.asarray(ds["edge_target"].values, dtype=np.int64),
            "node_y": np.asarray(ds["node_y"].values, dtype=np.int64),
            "node_x": np.asarray(ds["node_x"].values, dtype=np.int64),
            "gauge_index": np.asarray(ds["gauge_index"].values, dtype=np.int64) if "gauge_index" in ds else np.empty(0, dtype=np.int64),
            "gauge_id": np.asarray(ds["gauge_id"].values).astype(str) if "gauge_id" in ds else np.empty(0, dtype=str),
            "attrs": dict(ds.attrs),
        }
        return graph
    finally:
        ds.close()


def _graph_node_coordinates(graph: dict[str, np.ndarray | dict], x2d: np.ndarray, y2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_y = graph["node_y"]
    node_x = graph["node_x"]
    return x2d[node_y, node_x], y2d[node_y, node_x]


def _graph_segments(graph: dict[str, np.ndarray | dict], node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
    source = graph["edge_source"]
    target = graph["edge_target"]
    return np.stack(
        [
            np.stack([node_x[source], node_y[source]], axis=-1),
            np.stack([node_x[target], node_y[target]], axis=-1),
        ],
        axis=1,
    )


def _plot_dem(ax, dem: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> None:
    ax.pcolormesh(x2d, y2d, np.where(np.isfinite(dem), dem, np.nan), cmap="Greys", shading="nearest", alpha=0.20)


def _plot_gauges(ax, gauge_metadata: Path | None, graph: dict[str, np.ndarray | dict]) -> None:
    if gauge_metadata is None or not gauge_metadata.is_file():
        return
    meta = pd.read_csv(gauge_metadata, dtype={"basin_id": str})
    if not {"basin_id", "x", "y"}.issubset(meta.columns):
        return
    gauge_ids = set(str(value) for value in graph.get("gauge_id", []))
    if gauge_ids:
        meta = meta[meta["basin_id"].isin(gauge_ids)]
    ax.scatter(meta["x"], meta["y"], s=18, c="#f03b20", marker="x", linewidth=0.8, label="gauges", zorder=8)
    for row in meta.itertuples(index=False):
        ax.text(float(row.x), float(row.y), str(row.basin_id), fontsize=4.8, color="#252525", zorder=9)


def main() -> None:
    args = _parse_args()
    network_files = _discover_network_files(args.network, args.network_pattern, args.outlet_gauges)
    network = _read_flowpaths(network_files, args.flowpath_layer)
    dem, x2d, y2d, dem_crs = _load_dem(args.dem, args.dem_variable)
    if dem_crs is not None and network.crs is not None and not args.no_reproject_network:
        network = network.to_crs(dem_crs)
    graph = _load_graph(args.graph)
    node_x, node_y = _graph_node_coordinates(graph, x2d, y2d)
    segments = _graph_segments(graph, node_x, node_y)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), sharex=True, sharey=True)
    for ax in axes:
        _plot_dem(ax, dem, x2d, y2d)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x projection [m]")
        ax.set_ylabel("y projection [m]")

    network.plot(ax=axes[0], color="#238b45", linewidth=args.network_width, alpha=0.85)
    _plot_gauges(axes[0], args.gauge_metadata, graph)
    axes[0].set_title(
        "Ngen Source River Network: flowpaths layer\n"
        f"files={len(network_files)}, unique flowpaths={len(network)}"
    )

    axes[1].add_collection(LineCollection(segments, colors="#225ea8", linewidths=args.edge_width, alpha=args.edge_alpha, zorder=4))
    axes[1].scatter(node_x, node_y, s=1.0, c="#08306b", alpha=0.28, zorder=5, label="graph nodes")
    if len(graph["gauge_index"]):
        gauge_x = node_x[graph["gauge_index"]]
        gauge_y = node_y[graph["gauge_index"]]
        axes[1].scatter(gauge_x, gauge_y, s=22, c="#ff7f00", edgecolor="black", linewidth=0.25, zorder=7, label="gauge nodes")
    _plot_gauges(axes[1], args.gauge_metadata, graph)

    attrs = graph["attrs"]
    component_count = attrs.get("component_count", "unknown")
    outlet_gauges = attrs.get("outlet_gauges", "")
    try:
        outlet_gauges = ", ".join(json.loads(outlet_gauges)) if isinstance(outlet_gauges, str) else str(outlet_gauges)
    except Exception:
        outlet_gauges = str(outlet_gauges)
    axes[1].set_title(
        "Generated GNN Routing Graph\n"
        f"nodes={len(node_x)}, edges={len(graph['edge_source'])}, components={component_count}, outlets={outlet_gauges}"
    )

    xmin, xmax = float(np.nanmin(x2d)), float(np.nanmax(x2d))
    ymin, ymax = float(np.nanmin(y2d)), float(np.nanmax(y2d))
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left", frameon=True)

    fig.suptitle("Ngen River Network vs GNN Routing Graph", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
