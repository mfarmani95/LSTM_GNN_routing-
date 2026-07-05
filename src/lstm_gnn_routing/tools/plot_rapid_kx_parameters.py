"""Plot RAPID Muskingum K and X parameters for river reaches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def _read_single_column(path: Path, name: str, dtype=None) -> pd.Series:
    return pd.read_csv(path, header=None, names=[name], dtype=dtype)[name]


def _read_coords(path: Path) -> pd.DataFrame:
    coords = pd.read_csv(path)
    cols = {col.lower(): col for col in coords.columns}
    rivid_col = cols.get("rivid") or cols.get("comid") or cols.get("river_id")
    lat_col = cols.get("lat") or cols.get("latitude")
    lon_col = cols.get("lon") or cols.get("longitude")
    if rivid_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"{path} must contain rivid/comid, lat, and lon columns.")
    coords = coords.rename(columns={rivid_col: "rivid", lat_col: "lat", lon_col: "lon"})
    coords["rivid"] = coords["rivid"].astype(np.int64)
    coords = coords[
        coords["lat"].between(-90.0, 90.0)
        & coords["lon"].between(-180.0, 180.0)
        & np.isfinite(coords["lat"])
        & np.isfinite(coords["lon"])
    ].copy()
    # RIVID files can contain multiple forcing-grid cells per river.  Use a
    # centroid-like mean coordinate for reach-level visualization.
    return coords.groupby("rivid", as_index=False)[["lat", "lon"]].mean()


def _read_rapid_parameters(input_dir: Path, coords_file: Path) -> pd.DataFrame:
    rivid = _read_single_column(input_dir / "riv_bas_id_SRP.csv", "rivid", dtype=np.int64)
    k = _read_single_column(input_dir / "K_SRP.csv", "K")
    x = _read_single_column(input_dir / "X_SRP.csv", "X")
    if not (len(rivid) == len(k) == len(x)):
        raise ValueError(
            f"RAPID K/X/rivid row counts differ: rivid={len(rivid)}, K={len(k)}, X={len(x)}"
        )
    params = pd.DataFrame({"row": np.arange(len(rivid)), "rivid": rivid, "K": k.astype(float), "X": x.astype(float)})
    coords = _read_coords(coords_file)
    return params.merge(coords, on="rivid", how="left")


def _read_connectivity(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "rapid_connect_SRP.csv"
    conn = pd.read_csv(path, header=None)
    if conn.shape[1] < 2:
        raise ValueError(f"{path} must have at least source and downstream columns.")
    conn = conn.rename(columns={0: "rivid", 1: "downstream_rivid"})
    conn["rivid"] = conn["rivid"].astype(np.int64)
    conn["downstream_rivid"] = conn["downstream_rivid"].astype(np.int64)
    return conn[["rivid", "downstream_rivid"]]


def _plot_boundary(ax: plt.Axes, boundary_file: Path | None) -> None:
    if boundary_file is None:
        return
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        print(f"Could not import geopandas; skipped boundary overlay ({exc}).")
        return
    gdf = gpd.read_file(boundary_file)
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, alpha=0.9, zorder=5)


def _make_segments(params: pd.DataFrame, connectivity: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loc = params.set_index("rivid")[["lon", "lat", "K", "X"]]
    rows = []
    values_k = []
    values_x = []
    for row in connectivity.itertuples(index=False):
        if int(row.downstream_rivid) == 0:
            continue
        if row.rivid not in loc.index or row.downstream_rivid not in loc.index:
            continue
        p0 = loc.loc[row.rivid]
        p1 = loc.loc[row.downstream_rivid]
        if not np.all(np.isfinite([p0.lon, p0.lat, p1.lon, p1.lat])):
            continue
        rows.append([[float(p0.lon), float(p0.lat)], [float(p1.lon), float(p1.lat)]])
        values_k.append(float(p0.K))
        values_x.append(float(p0.X))
    return np.asarray(rows, dtype=float), np.asarray(values_k, dtype=float), np.asarray(values_x, dtype=float)


def _plot_parameter_map(
    output_path: Path,
    params: pd.DataFrame,
    segments: np.ndarray,
    segment_values: np.ndarray,
    parameter: str,
    *,
    boundary_file: Path | None = None,
    log_color: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    if len(segments):
        values = segment_values
        label = parameter
        if log_color:
            values = np.log10(np.clip(values, 1e-12, None))
            label = f"log10({parameter})"
        collection = LineCollection(segments, array=values, cmap="jet", linewidths=0.7, alpha=0.85)
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.84)
        cbar.set_label(label)
    else:
        values = params[parameter].to_numpy(dtype=float)
        if log_color:
            values = np.log10(np.clip(values, 1e-12, None))
        scatter = ax.scatter(params["lon"], params["lat"], c=values, s=4, cmap="jet", alpha=0.7)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.84)
        cbar.set_label(f"log10({parameter})" if log_color else parameter)
    _plot_boundary(ax, boundary_file)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"RAPID Muskingum {parameter} by River Reach")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_histograms(output_path: Path, params: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(params["K"].dropna(), bins=50, color="#2f6f8f", alpha=0.85)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Reach count")
    axes[0].set_title("RAPID K distribution")
    axes[1].hist(params["X"].dropna(), bins=30, color="#b12a1c", alpha=0.85)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Reach count")
    axes[1].set_title("RAPID X distribution")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _clip_to_boundary(gdf, boundary_file: Path | None):
    if boundary_file is None:
        return gdf
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        print(f"Could not import geopandas; skipped clipping ({exc}).")
        return gdf
    boundary = gpd.read_file(boundary_file)
    if boundary.crs is not None and gdf.crs is not None and boundary.crs != gdf.crs:
        boundary = boundary.to_crs(gdf.crs)
    geom = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.geometry.unary_union
    clipped = gdf[gdf.geometry.intersects(geom)].copy()
    clipped["geometry"] = clipped.geometry.intersection(geom)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    return clipped


def _plot_flowline_parameter_map(
    output_path: Path,
    flowlines,
    parameter: str,
    *,
    boundary_file: Path | None = None,
    log_color: bool = False,
) -> None:
    plot_gdf = flowlines.copy()
    column = parameter
    legend_label = parameter
    if log_color:
        column = f"log10_{parameter}"
        plot_gdf[column] = np.log10(np.clip(plot_gdf[parameter].astype(float), 1e-12, None))
        legend_label = f"log10({parameter})"
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    plot_gdf.plot(
        ax=ax,
        column=column,
        cmap="jet",
        linewidth=0.8,
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.84},
    )
    _plot_boundary(ax, boundary_file)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"RAPID Muskingum {parameter} by Flowline COMID")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _read_flowline_parameters(flowline_file: Path, params: pd.DataFrame, boundary_file: Path | None = None):
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("geopandas is required for --flowline-file plotting") from exc
    flowlines = gpd.read_file(flowline_file)
    comid_col = next((col for col in flowlines.columns if col.lower() in {"comid", "rivid"}), None)
    if comid_col is None:
        raise ValueError(f"{flowline_file} does not contain a COMID/rivid column.")
    flowlines = flowlines.rename(columns={comid_col: "COMID"})
    flowlines["COMID"] = flowlines["COMID"].astype(np.int64)
    joined = flowlines.merge(params[["rivid", "K", "X"]], left_on="COMID", right_on="rivid", how="inner")
    if len(joined) == 0:
        raise ValueError("No flowline COMIDs matched RAPID K/X rivid values.")
    return _clip_to_boundary(joined, boundary_file)


def _print_summary(params: pd.DataFrame) -> None:
    print("Joined RAPID reaches:", len(params))
    print("Missing coordinates:", int(params[["lat", "lon"]].isna().any(axis=1).sum()))
    print(params[["K", "X"]].describe().to_string())
    print("Unique X values:", params["X"].nunique(dropna=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/xdisk/niug/farmani/rapid_imput_srp")
    parser.add_argument(
        "--coords-file",
        default="/xdisk/niug/farmani/RAPID/inflow/workflow/inflow_yml/RIVID/RIVID_SRP.csv",
        help="CSV containing rivid/comid, lat, and lon columns.",
    )
    parser.add_argument("--boundary-file", default=None, help="Optional basin boundary shapefile/GeoJSON.")
    parser.add_argument(
        "--flowline-file",
        default=None,
        help="Optional flowline shapefile with COMID geometry. If provided, maps use true flowline geometry.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--linear-k", action="store_true", help="Use linear K colors instead of log10(K).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    coords_file = Path(args.coords_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "plots_kx"
    boundary_file = Path(args.boundary_file).expanduser().resolve() if args.boundary_file else None
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _read_rapid_parameters(input_dir, coords_file)
    connectivity = _read_connectivity(input_dir)
    segments, segment_k, segment_x = _make_segments(params, connectivity)
    params.to_csv(output_dir / "rapid_k_x_reach_parameters.csv", index=False)
    if args.flowline_file:
        flowline_file = Path(args.flowline_file).expanduser().resolve()
        flowlines = _read_flowline_parameters(flowline_file, params, boundary_file=boundary_file)
        flowlines.drop(columns="geometry").to_csv(output_dir / "rapid_k_x_flowline_attributes.csv", index=False)
        flowlines.to_file(output_dir / "rapid_k_x_flowlines.gpkg", driver="GPKG")
        _plot_flowline_parameter_map(
            output_dir / "rapid_K_map.png",
            flowlines,
            "K",
            boundary_file=boundary_file,
            log_color=not args.linear_k,
        )
        _plot_flowline_parameter_map(
            output_dir / "rapid_X_map.png",
            flowlines,
            "X",
            boundary_file=boundary_file,
            log_color=False,
        )
        print(f"Flowlines plotted after join/clip: {len(flowlines)}")
    else:
        _plot_parameter_map(
            output_dir / "rapid_K_map.png",
            params,
            segments,
            segment_k,
            "K",
            boundary_file=boundary_file,
            log_color=not args.linear_k,
        )
        _plot_parameter_map(
            output_dir / "rapid_X_map.png",
            params,
            segments,
            segment_x,
            "X",
            boundary_file=boundary_file,
            log_color=False,
        )
    _plot_histograms(output_dir / "rapid_K_X_histograms.png", params)
    _print_summary(params)
    print(f"Saved RAPID K/X plots to {output_dir}")


if __name__ == "__main__":
    main()
