from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_GRAPH = Path("Pytorch/routing_graph_ngen_salt_verde_cache.nc")
DEFAULT_DEM = Path("Input/SRTM/basin_srtm_dem_conditioned_on_forcing_grid.nc")
DEFAULT_GAUGES = Path("Input/Streamflow/30_gauges_IN_LAMBERT.csv")
DEFAULT_OUTPUT_DIR = Path("Pytorch/output/ngen_qc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QC maps for an Ngen routing graph cache.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--dem-variable", default="DEM")
    parser.add_argument("--gauge-metadata", type=Path, default=DEFAULT_GAUGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def _load_dem(path: Path, variable: str):
    ds = xr.open_dataset(path)
    try:
        dem = np.asarray(ds[variable].values, dtype=np.float32)
        x_values = np.asarray(ds["x"].values)
        y_values = np.asarray(ds["y"].values)
        if x_values.ndim == 1 and y_values.ndim == 1:
            x2d, y2d = np.meshgrid(x_values, y_values, indexing="xy")
        else:
            x2d, y2d = x_values, y_values
        return dem, x2d, y2d
    finally:
        ds.close()


def _load_graph(path: Path) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(path)
    try:
        graph = {
            "edge_source": np.asarray(ds["edge_source"].values, dtype=np.int64),
            "edge_target": np.asarray(ds["edge_target"].values, dtype=np.int64),
            "node_y": np.asarray(ds["node_y"].values, dtype=np.int64),
            "node_x": np.asarray(ds["node_x"].values, dtype=np.int64),
            "flat_index": np.asarray(ds["flat_index"].values, dtype=np.int64),
            "runoff_source_flat_index": np.asarray(ds["runoff_source_flat_index"].values, dtype=np.int64),
            "gauge_index": np.asarray(ds["gauge_index"].values, dtype=np.int64),
            "gauge_id": np.asarray(ds["gauge_id"].values).astype(str),
            "attrs": dict(ds.attrs),
        }
        return graph
    finally:
        ds.close()


def _axis_limits(x2d: np.ndarray, y2d: np.ndarray):
    return float(np.nanmin(x2d)), float(np.nanmax(x2d)), float(np.nanmin(y2d)), float(np.nanmax(y2d))


def _plot_dem_mapping_qc(dem: np.ndarray, x2d: np.ndarray, y2d: np.ndarray, graph: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    active = np.isfinite(dem)
    mapped = np.zeros(active.size, dtype=bool)
    mapped[graph["runoff_source_flat_index"]] = True
    mapped = mapped.reshape(active.shape)
    unmapped = active & ~mapped

    fig, ax = plt.subplots(figsize=(13, 9))
    image = np.full(active.shape, np.nan, dtype=np.float32)
    image[active] = 1.0
    image[mapped] = 2.0
    image[unmapped] = 3.0
    cmap = plt.matplotlib.colors.ListedColormap(["#d9d9d9", "#7fc97f", "#f0027f"])
    norm = plt.matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.pcolormesh(x2d, y2d, image, cmap=cmap, norm=norm, shading="nearest", alpha=0.82)
    ax.set_title(
        "DEM Active Cells vs Ngen Divide-Mapped Runoff Sources\n"
        f"active={int(active.sum())}, mapped={int(mapped.sum())}, unmapped={int(unmapped.sum())}"
    )
    ax.set_xlabel("x projection [m]")
    ax.set_ylabel("y projection [m]")
    ax.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#d9d9d9", markersize=10, label="DEM active only"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#7fc97f", markersize=10, label="Mapped to Ngen divide"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f0027f", markersize=10, label="DEM active but unmapped"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_graph_qc(
    dem: np.ndarray,
    x2d: np.ndarray,
    y2d: np.ndarray,
    graph: dict[str, np.ndarray],
    gauge_metadata: Path,
    output: Path,
    dpi: int,
) -> None:
    node_x = x2d[graph["node_y"], graph["node_x"]]
    node_y = y2d[graph["node_y"], graph["node_x"]]
    source = graph["edge_source"]
    target = graph["edge_target"]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.pcolormesh(x2d, y2d, np.where(np.isfinite(dem), dem, np.nan), cmap="terrain", shading="nearest", alpha=0.35)

    for src, dst in zip(source, target):
        ax.plot(
            [node_x[src], node_x[dst]],
            [node_y[src], node_y[dst]],
            color="#225ea8",
            linewidth=0.25,
            alpha=0.35,
            zorder=2,
        )
    ax.scatter(node_x, node_y, s=1.5, c="#08519c", alpha=0.35, label="Flowpath graph nodes", zorder=3)

    gauge_node_x = node_x[graph["gauge_index"]]
    gauge_node_y = node_y[graph["gauge_index"]]
    ax.scatter(gauge_node_x, gauge_node_y, s=35, c="#ff7f00", edgecolor="black", linewidth=0.4, label="Gauge graph nodes", zorder=5)

    if gauge_metadata.is_file():
        meta = pd.read_csv(gauge_metadata, dtype={"basin_id": str})
        meta = meta[meta["basin_id"].isin(set(graph["gauge_id"].tolist()))]
        if {"x", "y"}.issubset(meta.columns):
            ax.scatter(meta["x"], meta["y"], s=18, c="red", marker="x", linewidth=0.8, label="Gauge metadata x/y", zorder=6)
        for row in meta.itertuples(index=False):
            ax.text(float(row.x), float(row.y), str(row.basin_id), fontsize=5.5, color="black", zorder=7)

    xmin, xmax, ymin, ymax = _axis_limits(x2d, y2d)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x projection [m]")
    ax.set_ylabel("y projection [m]")
    ax.set_title(
        "Generated Ngen Routing Graph on DEM Grid\n"
        f"nodes={len(node_x)}, edges={len(source)}, gauges={len(graph['gauge_id'])}, "
        f"components={graph['attrs'].get('component_count', 'unknown')}"
    )
    ax.legend(loc="lower left", frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    dem, x2d, y2d = _load_dem(args.dem, args.dem_variable)
    graph = _load_graph(args.graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_dem_mapping_qc(
        dem,
        x2d,
        y2d,
        graph,
        args.output_dir / "ngen_dem_mapping_qc.png",
        args.dpi,
    )
    _plot_graph_qc(
        dem,
        x2d,
        y2d,
        graph,
        args.gauge_metadata,
        args.output_dir / "ngen_routing_graph_qc.png",
        args.dpi,
    )
    print(f"Wrote {args.output_dir / 'ngen_dem_mapping_qc.png'}")
    print(f"Wrote {args.output_dir / 'ngen_routing_graph_qc.png'}")


if __name__ == "__main__":
    main()
