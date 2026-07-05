from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr

from lstm_gnn_routing.tools.analyze_rapid_vs_gnn import (
    _apply_map_bounds,
    _background_bounds,
    _infer_comids_from_nldi,
    _infer_comids_from_rapid_lonlat,
    _plot_background_shape,
    _prepare_background_shape,
    _read_metadata,
    _read_rapid_timeseries,
    _read_background_shape,
    _resolve_plot_crs,
)
from lstm_gnn_routing.tools.rapid_file import detect_rapid_file


DAM_FILTERED_GAUGES = [
    "09489500",
    "09490500",
    "09497500",
    "09492400",
    "09496500",
    "09494000",
    "09498500",
    "09498400",
    "09497980",
    "09497800",
    "09499000",
    "09503700",
    "09504000",
    "09504420",
    "09504500",
    "09505350",
    "09505200",
    "09505800",
    "09506000",
    "09507980",
    "09508300",
    "09508500",
    "09510200",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gauge-level mass-balance diagnostics for continuous GNN routing output.")
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--period", default="test")
    parser.add_argument("--graph-file", required=True, type=Path)
    parser.add_argument("--forcing-dir", required=True, type=Path)
    parser.add_argument("--rapid-file", default=None, type=Path)
    parser.add_argument("--gauge-metadata", required=True, type=Path)
    parser.add_argument("--background-shapefile", type=Path, default=Path("data/HUC4/Salt_and_Verde.shp"))
    parser.add_argument("--nldi-comid-cache", type=Path, default=Path("data/streamflow/usgs_nldi_comids.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stations", nargs="+", default=["dam_filtered"])
    parser.add_argument("--runoff-vars", nargs="+", default=["RUNSF", "RUNSB"])
    parser.add_argument("--qout-var", default="Qout")
    parser.add_argument("--rivid-dim", default=None)
    parser.add_argument("--time-dim", default=None)
    parser.add_argument("--time-dim-forcing", default="time")
    parser.add_argument("--y-dim", default="y")
    parser.add_argument("--x-dim", default="x")
    parser.add_argument("--trim-days", type=int, default=30)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--season-windows", nargs="*", default=["Dec-May:12,1,2,3,4,5", "Jun-Sep:6,7,8,9"])
    parser.add_argument("--infer-comids-from-nldi", action="store_true")
    parser.add_argument("--infer-comids-from-rapid-lonlat", action="store_true")
    parser.add_argument("--nldi-timeout", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _read_gnn_timeseries(evaluation_dir: Path, period: str, stations: list[str]) -> pd.DataFrame:
    path = evaluation_dir / f"{period}_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GNN timeseries: {path}")
    ts = pd.read_csv(path, dtype={"gauge_id": str}, parse_dates=["date"])
    ts = ts[ts["gauge_id"].isin(set(stations))].copy()
    if ts.empty:
        raise ValueError(f"No requested stations were found in {path}")
    ts = ts.rename(columns={"prediction": "gnn", "observation": "observed"})
    return ts[["gauge_id", "date", "gnn", "observed"]]


def _resolve_stations(requested: list[str], evaluation_dir: Path, period: str) -> list[str]:
    if len(requested) == 1 and requested[0] == "dam_filtered":
        return DAM_FILTERED_GAUGES
    if len(requested) == 1 and requested[0] == "all_evaluated":
        path = evaluation_dir / f"{period}_timeseries.csv"
        ts = pd.read_csv(path, dtype={"gauge_id": str}, usecols=["gauge_id"])
        return sorted(ts["gauge_id"].dropna().astype(str).unique().tolist())
    return [str(value) for value in requested]


def _decode_names(values: Any) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _load_graph(graph_file: Path) -> dict[str, Any]:
    ds = xr.open_dataset(graph_file)
    graph = {name: ds[name].values for name in ds.data_vars}
    graph["attrs"] = dict(ds.attrs)
    ds.close()
    return graph


def _upstream_node_sets(edge_source: np.ndarray, edge_target: np.ndarray, gauge_nodes: dict[str, int]) -> dict[str, set[int]]:
    reverse_adj: dict[int, list[int]] = defaultdict(list)
    for src, dst in zip(edge_source.astype(int), edge_target.astype(int)):
        reverse_adj[int(dst)].append(int(src))
    result: dict[str, set[int]] = {}
    for gauge_id, gauge_node in gauge_nodes.items():
        seen = {int(gauge_node)}
        queue: deque[int] = deque([int(gauge_node)])
        while queue:
            node = queue.popleft()
            for upstream in reverse_adj.get(node, []):
                if upstream not in seen:
                    seen.add(upstream)
                    queue.append(upstream)
        result[gauge_id] = seen
    return result


def _source_to_gauge_matrix(graph: dict[str, Any], stations: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    gauge_ids = _decode_names(graph["gauge_id"])
    gauge_index = np.asarray(graph["gauge_index"], dtype=np.int64)
    gauge_node_by_id = {gauge_id: int(node) for gauge_id, node in zip(gauge_ids, gauge_index)}
    missing = [station for station in stations if station not in gauge_node_by_id]
    if missing:
        raise KeyError(f"Stations missing from graph gauge_id: {missing}")
    selected_nodes = {station: gauge_node_by_id[station] for station in stations}
    upstream = _upstream_node_sets(
        np.asarray(graph["edge_source"], dtype=np.int64),
        np.asarray(graph["edge_target"], dtype=np.int64),
        selected_nodes,
    )
    target_index = np.asarray(graph["runoff_target_index"], dtype=np.int64)
    source_flat = np.asarray(graph["runoff_source_flat_index"], dtype=np.int64)
    weights = np.asarray(graph["runoff_source_weight"], dtype=np.float64)
    membership = np.zeros((len(source_flat), len(stations)), dtype=np.float32)
    for j, station in enumerate(stations):
        nodes = upstream[station]
        membership[:, j] = np.isin(target_index, list(nodes)).astype(np.float32)
    return source_flat, weights, membership, selected_nodes


def _forcing_year_paths(forcing_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    paths = []
    for year in range(int(start.year), int(end.year) + 1):
        path = forcing_dir / f"{year}.zarr"
        if path.exists():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No yearly Zarr stores found in {forcing_dir} for {start.year}-{end.year}")
    return paths


def _compute_daily_upstream_runoff(
    *,
    forcing_dir: Path,
    runoff_vars: list[str],
    source_flat: np.ndarray,
    weights_m2: np.ndarray,
    membership: np.ndarray,
    stations: list[str],
    dates: pd.DatetimeIndex,
    time_dim: str,
    y_dim: str,
    x_dim: str,
) -> pd.DataFrame:
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    requested_dates = pd.DatetimeIndex(dates.normalize().unique())
    rows = []
    for path in _forcing_year_paths(forcing_dir, start, end):
        ds = xr.open_zarr(path, consolidated=True)
        try:
            ds = ds.sel({time_dim: slice(start, end)})
            if ds.sizes.get(time_dim, 0) == 0:
                continue
            time_values = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values).normalize())
            keep = time_values.isin(requested_dates)
            if not bool(np.any(keep)):
                continue
            ds = ds.isel({time_dim: np.where(keep)[0]})
            time_values = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values).normalize())
            ny = int(ds.sizes[y_dim])
            nx = int(ds.sizes[x_dim])
            yy = source_flat // nx
            xx = source_flat % nx
            if int(yy.max()) >= ny:
                raise ValueError(f"Source flat indices exceed forcing grid shape ({ny}, {nx}) in {path}")
            runoff = None
            for var in runoff_vars:
                values = np.asarray(ds[var].values[:, yy, xx], dtype=np.float64)
                runoff = values if runoff is None else runoff + values
            # runoff is mm/day. weights are contributing m2. result is m3/day.
            source_volume = runoff * weights_m2.reshape(1, -1) / 1000.0
            # A small number of inactive or ocean-edge source cells can be NaN.
            # They should contribute no volume; otherwise NaN * 0 still poisons
            # every gauge sum in the dense matrix multiply below.
            source_volume = np.nan_to_num(source_volume, nan=0.0, posinf=0.0, neginf=0.0)
            gauge_volume = source_volume @ membership
            for t_index, date in enumerate(time_values):
                for g_index, station in enumerate(stations):
                    rows.append(
                        {
                            "gauge_id": station,
                            "date": date,
                            "runoff_m3_day": float(gauge_volume[t_index, g_index]),
                        }
                    )
        finally:
            ds.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No runoff volumes were computed.")
    return frame.sort_values(["gauge_id", "date"]).reset_index(drop=True)


def _parse_season_windows(specs: list[str]) -> dict[str, set[int]]:
    seasons = {}
    for spec in specs:
        name, months = spec.split(":", 1)
        seasons[name] = {int(value) for value in months.split(",") if value.strip()}
    return seasons


def _add_volume_columns(aligned: pd.DataFrame) -> pd.DataFrame:
    out = aligned.copy()
    out["gnn_m3_day"] = out["gnn"] * 86400.0
    out["observed_m3_day"] = out["observed"] * 86400.0
    out["rapid_m3_day"] = out["rapid"] * 86400.0
    out["gnn_residual_m3_day"] = out["gnn_m3_day"] - out["runoff_m3_day"]
    out["rapid_residual_m3_day"] = out["rapid_m3_day"] - out["runoff_m3_day"]
    out["observed_residual_m3_day"] = out["observed_m3_day"] - out["runoff_m3_day"]
    return out


def _volume_summary(data: pd.DataFrame, group_cols: list[str], label_col: str | None = None) -> pd.DataFrame:
    grouped = data.groupby(group_cols, as_index=False)[
        ["runoff_m3_day", "gnn_m3_day", "rapid_m3_day", "observed_m3_day"]
    ].sum()
    grouped = grouped.rename(
        columns={
            "runoff_m3_day": "runoff_volume_m3",
            "gnn_m3_day": "gnn_volume_m3",
            "rapid_m3_day": "rapid_volume_m3",
            "observed_m3_day": "observed_volume_m3",
        }
    )
    denom = grouped["runoff_volume_m3"].replace(0.0, np.nan)
    for source in ["gnn", "rapid", "observed"]:
        grouped[f"{source}_runoff_ratio"] = grouped[f"{source}_volume_m3"] / denom
        grouped[f"{source}_volume_bias_fraction"] = (grouped[f"{source}_volume_m3"] - grouped["runoff_volume_m3"]) / denom
    if label_col and label_col not in grouped:
        grouped[label_col] = ""
    return grouped


def _monthly_volume_ratios(data: pd.DataFrame) -> pd.DataFrame:
    monthly = data.copy()
    monthly["month_start"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    grouped = _volume_summary(monthly, ["gauge_id", "month_start"])
    return grouped.sort_values(["gauge_id", "month_start"]).reset_index(drop=True)


def _plot_ratio_bars(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    frame = summary.sort_values("gauge_id")
    x = np.arange(len(frame))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(frame) * 0.45), 5))
    ax.bar(x - width, frame["gnn_runoff_ratio"], width, label="GNN", color="#c44900")
    ax.bar(x, frame["rapid_runoff_ratio"], width, label="RAPID", color="#2a9d8f")
    ax.bar(x + width, frame["observed_runoff_ratio"], width, label="Observed", color="#1f4e79")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["gauge_id"], rotation=70, ha="right")
    ax.set_ylabel("Streamflow volume / upstream runoff volume")
    ax.set_title("Full-period gauge water-volume ratio")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bar_volume_ratio_by_gauge.png", dpi=dpi)
    plt.close(fig)


def _plot_scatter(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    max_value = float(summary[["runoff_volume_m3", "gnn_volume_m3", "rapid_volume_m3", "observed_volume_m3"]].max().max())
    ax.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1.0, label="1:1")
    ax.scatter(summary["runoff_volume_m3"], summary["gnn_volume_m3"], label="GNN", color="#c44900", alpha=0.8)
    ax.scatter(summary["runoff_volume_m3"], summary["rapid_volume_m3"], label="RAPID", color="#2a9d8f", alpha=0.8)
    ax.scatter(summary["runoff_volume_m3"], summary["observed_volume_m3"], label="Observed", color="#1f4e79", alpha=0.8)
    ax.set_xlabel("Upstream runoff volume (m3)")
    ax.set_ylabel("Streamflow volume (m3)")
    ax.set_title("Gauge volume closure")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_runoff_vs_streamflow_volume.png", dpi=dpi)
    plt.close(fig)


def _plot_monthly_ratio_timeseries(monthly: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ratio_dir = output_dir / "monthly_ratio_timeseries_by_gauge"
    ratio_dir.mkdir(parents=True, exist_ok=True)
    for gauge_id, group in monthly.groupby("gauge_id", sort=True):
        group = group.sort_values("month_start")
        fig, ax = plt.subplots(figsize=(12.5, 4.6))
        ax.plot(
            group["month_start"],
            group["gnn_runoff_ratio"],
            label="GNN / runoff",
            color="#c44900",
            linewidth=1.25,
            alpha=0.9,
        )
        ax.plot(
            group["month_start"],
            group["rapid_runoff_ratio"],
            label="RAPID / runoff",
            color="#2a9d8f",
            linewidth=1.25,
            alpha=0.9,
        )
        ax.plot(
            group["month_start"],
            group["observed_runoff_ratio"],
            label="Observed / runoff",
            color="#1f4e79",
            linewidth=1.25,
            alpha=0.9,
        )
        ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", label="1:1 volume")
        ax.set_title(f"Monthly streamflow/runoff volume ratio: {gauge_id}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly streamflow volume / upstream runoff volume")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(ratio_dir / f"{gauge_id}_monthly_streamflow_runoff_ratio.png", dpi=dpi)
        plt.close(fig)


def _plot_cumulative(aligned: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    cumulative_dir = output_dir / "cumulative_residual_by_gauge"
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    for gauge_id, group in aligned.groupby("gauge_id", sort=True):
        group = group.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(group["date"], group["gnn_residual_m3_day"].cumsum(), label="GNN - runoff", color="#c44900")
        ax.plot(group["date"], group["rapid_residual_m3_day"].cumsum(), label="RAPID - runoff", color="#2a9d8f")
        ax.plot(group["date"], group["observed_residual_m3_day"].cumsum(), label="Observed - runoff", color="#1f4e79")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Cumulative mass residual: {gauge_id}")
        ax.set_ylabel("Cumulative residual (m3)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(cumulative_dir / f"{gauge_id}_cumulative_residual.png", dpi=dpi)
        plt.close(fig)


def _plot_heatmap(table: pd.DataFrame, value_col: str, output: Path, title: str, dpi: int, vmin=None, vmax=None) -> None:
    pivot = table.pivot(index="gauge_id", columns="year", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(max(10, 0.4 * len(pivot.columns)), max(6, 0.28 * len(pivot.index))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_maps(summary: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path, dpi: int, background_shape) -> None:
    merged = summary.merge(metadata, on="gauge_id", how="left")
    # Mass-balance maps are intentionally plotted in geographic lon/lat, so
    # the gauge points and Salt-Verde boundary use the same visual reference.
    x_col, y_col = ("lon", "lat")
    if not {x_col, y_col}.issubset(merged.columns):
        return
    target_crs = _resolve_plot_crs((x_col, y_col), None, None)
    plot_background = _prepare_background_shape(background_shape, target_crs)
    background_bounds = _background_bounds(plot_background)
    ratio_norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    for value_col, title, filename, norm, cmap, colorbar_label in [
        (
            "gnn_runoff_ratio",
            "GNN streamflow/runoff volume ratio",
            "map_gnn_runoff_ratio.png",
            ratio_norm,
            "RdBu",
            "Streamflow/runoff ratio (0-2, white=1)",
        ),
        (
            "rapid_runoff_ratio",
            "RAPID streamflow/runoff volume ratio",
            "map_rapid_runoff_ratio.png",
            ratio_norm,
            "RdBu",
            "Streamflow/runoff ratio (0-2, white=1)",
        ),
        (
            "observed_runoff_ratio",
            "Observed streamflow/runoff volume ratio",
            "map_observed_runoff_ratio.png",
            ratio_norm,
            "RdBu",
            "Streamflow/runoff ratio (0-2, white=1)",
        ),
        (
            "gnn_minus_rapid_ratio",
            "GNN - RAPID volume ratio",
            "map_gnn_minus_rapid_volume_ratio.png",
            TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5),
            "RdBu",
            "GNN - RAPID ratio",
        ),
    ]:
        if value_col not in merged:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_background_shape(ax, plot_background)
        sc = ax.scatter(
            merged[x_col],
            merged[y_col],
            c=merged[value_col],
            s=38,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        _apply_map_bounds(ax, background_bounds)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi)
        plt.close(fig)


def _gauge_to_gauge_topology(
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    gauge_nodes: dict[str, int],
) -> pd.DataFrame:
    upstream_node_sets = _upstream_node_sets(edge_source, edge_target, gauge_nodes)
    all_upstream_by_gauge: dict[str, set[str]] = {}
    for gauge_id, nodes in upstream_node_sets.items():
        all_upstream_by_gauge[gauge_id] = {
            other_id
            for other_id, other_node in gauge_nodes.items()
            if other_id != gauge_id and int(other_node) in nodes
        }

    rows = []
    for gauge_id in sorted(gauge_nodes):
        all_upstream = all_upstream_by_gauge[gauge_id]
        frontier = []
        for candidate in sorted(all_upstream):
            # If candidate is upstream of another upstream gauge, summing both
            # double-counts the nested branch. Frontier is the nearest upstream
            # gauge cut-set and is the preferred streamflow-to-streamflow check.
            nested = any(
                candidate in all_upstream_by_gauge.get(other, set())
                for other in all_upstream
                if other != candidate
            )
            if not nested:
                frontier.append(candidate)
        rows.append(
            {
                "gauge_id": gauge_id,
                "gauge_node": int(gauge_nodes[gauge_id]),
                "frontier_upstream_gauge_count": len(frontier),
                "frontier_upstream_gauges": ",".join(frontier),
                "all_upstream_gauge_count": len(all_upstream),
                "all_upstream_gauges": ",".join(sorted(all_upstream)),
            }
        )
    return pd.DataFrame(rows)


def _gauge_to_gauge_source_table(
    aligned: pd.DataFrame,
    topology: pd.DataFrame,
    source: str,
    upstream_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_col = f"{source}_m3_day"
    upstream_col = f"{upstream_mode}_upstream_gauges"
    if value_col not in aligned:
        raise KeyError(f"Missing {value_col} in aligned mass-balance table.")
    if upstream_col not in topology:
        raise KeyError(f"Missing {upstream_col} in gauge topology table.")

    wide = aligned.pivot_table(index="date", columns="gauge_id", values=value_col, aggfunc="mean").sort_index()
    topology_by_gauge = topology.set_index("gauge_id")
    summary_rows = []
    monthly_rows = []
    for gauge_id, topo_row in topology_by_gauge.iterrows():
        upstream = [value for value in str(topo_row[upstream_col]).split(",") if value]
        upstream = [value for value in upstream if value in wide.columns]
        if not upstream or gauge_id not in wide.columns:
            continue

        downstream = wide[gauge_id]
        upstream_frame = wide[upstream]
        upstream_sum = upstream_frame.sum(axis=1, min_count=len(upstream))
        valid = downstream.notna() & upstream_frame.notna().all(axis=1) & upstream_sum.notna() & (upstream_sum != 0.0)
        if not bool(valid.any()):
            continue

        daily = pd.DataFrame(
            {
                "date": wide.index[valid],
                "downstream_volume_m3": downstream[valid].values,
                "upstream_gauge_volume_m3": upstream_sum[valid].values,
            }
        )
        daily["incremental_volume_m3"] = daily["downstream_volume_m3"] - daily["upstream_gauge_volume_m3"]
        daily["downstream_to_upstream_ratio"] = daily["downstream_volume_m3"] / daily["upstream_gauge_volume_m3"]
        daily["month_start"] = pd.to_datetime(daily["date"]).dt.to_period("M").dt.to_timestamp()

        downstream_total = float(daily["downstream_volume_m3"].sum())
        upstream_total = float(daily["upstream_gauge_volume_m3"].sum())
        incremental_total = float(daily["incremental_volume_m3"].sum())
        summary_rows.append(
            {
                "source": source.upper(),
                "upstream_mode": upstream_mode,
                "gauge_id": gauge_id,
                "upstream_gauge_count": len(upstream),
                "upstream_gauges": ",".join(upstream),
                "valid_days": int(len(daily)),
                "analysis_start": str(pd.to_datetime(daily["date"]).min().date()),
                "analysis_end": str(pd.to_datetime(daily["date"]).max().date()),
                "downstream_volume_m3": downstream_total,
                "upstream_gauge_volume_m3": upstream_total,
                "downstream_to_upstream_ratio": downstream_total / upstream_total if upstream_total else np.nan,
                "mean_daily_ratio": float(daily["downstream_to_upstream_ratio"].mean()),
                "median_daily_ratio": float(daily["downstream_to_upstream_ratio"].median()),
                "incremental_volume_m3": incremental_total,
                "incremental_fraction_of_downstream": incremental_total / downstream_total if downstream_total else np.nan,
            }
        )

        monthly = daily.groupby("month_start", as_index=False).agg(
            downstream_volume_m3=("downstream_volume_m3", "sum"),
            upstream_gauge_volume_m3=("upstream_gauge_volume_m3", "sum"),
            incremental_volume_m3=("incremental_volume_m3", "sum"),
            valid_days=("date", "count"),
        )
        monthly["downstream_to_upstream_ratio"] = (
            monthly["downstream_volume_m3"] / monthly["upstream_gauge_volume_m3"].replace(0.0, np.nan)
        )
        monthly["source"] = source.upper()
        monthly["upstream_mode"] = upstream_mode
        monthly["gauge_id"] = gauge_id
        monthly["upstream_gauge_count"] = len(upstream)
        monthly["upstream_gauges"] = ",".join(upstream)
        monthly_rows.append(monthly)

    return pd.DataFrame(summary_rows), pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()


def _gauge_to_gauge_diagnostics(
    aligned: pd.DataFrame,
    graph: dict[str, Any],
    gauge_nodes: dict[str, int],
    metadata: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    background_shape,
) -> None:
    g2g_dir = output_dir / "gauge_to_gauge_balance"
    g2g_dir.mkdir(parents=True, exist_ok=True)

    topology = _gauge_to_gauge_topology(
        np.asarray(graph["edge_source"], dtype=np.int64),
        np.asarray(graph["edge_target"], dtype=np.int64),
        gauge_nodes,
    )
    topology.to_csv(g2g_dir / "gauge_to_gauge_topology.csv", index=False)

    summary_frames = []
    monthly_frames = []
    for source in ["gnn", "rapid", "observed"]:
        for upstream_mode in ["frontier", "all"]:
            summary, monthly = _gauge_to_gauge_source_table(aligned, topology, source, upstream_mode)
            if not summary.empty:
                summary_frames.append(summary)
            if not monthly.empty:
                monthly_frames.append(monthly)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    if summary.empty:
        raise ValueError("Gauge-to-gauge diagnostic found no gauges with upstream gauge constraints.")

    summary.to_csv(g2g_dir / "gauge_to_gauge_balance_by_gauge.csv", index=False)
    monthly.to_csv(g2g_dir / "gauge_to_gauge_balance_by_gauge_month.csv", index=False)

    comparison = summary.pivot_table(
        index=["upstream_mode", "gauge_id"],
        columns="source",
        values="downstream_to_upstream_ratio",
        aggfunc="first",
    ).reset_index()
    for left, right in [("GNN", "RAPID"), ("GNN", "OBSERVED"), ("RAPID", "OBSERVED")]:
        if left in comparison and right in comparison:
            comparison[f"{left.lower()}_minus_{right.lower()}_ratio"] = comparison[left] - comparison[right]
    comparison.to_csv(g2g_dir / "gauge_to_gauge_ratio_comparison.csv", index=False)

    _plot_gauge_to_gauge_monthly(monthly, g2g_dir, dpi)
    _plot_gauge_to_gauge_boxplot(monthly, g2g_dir, dpi)
    _plot_gauge_to_gauge_maps(summary, comparison, metadata, g2g_dir, dpi, background_shape)


def _plot_gauge_to_gauge_monthly(monthly: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    plot_dir = output_dir / "monthly_ratio_timeseries_by_gauge"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"GNN": "#c44900", "RAPID": "#2a9d8f", "OBSERVED": "#1f4e79"}
    for (upstream_mode, gauge_id), group in monthly.groupby(["upstream_mode", "gauge_id"], sort=True):
        fig, ax = plt.subplots(figsize=(12.5, 4.6))
        for source, source_group in group.groupby("source", sort=True):
            source_group = source_group.sort_values("month_start")
            ax.plot(
                source_group["month_start"],
                source_group["downstream_to_upstream_ratio"],
                label=source,
                color=colors.get(source, None),
                linewidth=1.25,
                alpha=0.9,
            )
        ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", label="1:1")
        ax.set_title(f"Monthly downstream/upstream-gauge ratio: {gauge_id} ({upstream_mode})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Downstream streamflow / upstream-gauge streamflow")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{upstream_mode}_{gauge_id}_monthly_gauge_to_gauge_ratio.png", dpi=dpi)
        plt.close(fig)


def _plot_gauge_to_gauge_boxplot(monthly: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    for upstream_mode, mode_group in monthly.groupby("upstream_mode", sort=True):
        stations = sorted(mode_group["gauge_id"].unique())
        sources = [source for source in ["GNN", "RAPID", "OBSERVED"] if source in set(mode_group["source"])]
        x = np.arange(len(stations))
        width = 0.24
        fig, ax = plt.subplots(figsize=(max(12, len(stations) * 0.45), 5.8))
        for i, source in enumerate(sources):
            values = [
                mode_group[(mode_group["gauge_id"] == station) & (mode_group["source"] == source)][
                    "downstream_to_upstream_ratio"
                ].dropna()
                for station in stations
            ]
            ax.boxplot(
                values,
                positions=x + (i - (len(sources) - 1) / 2) * width,
                widths=width * 0.85,
                patch_artist=True,
                boxprops={"facecolor": f"C{i}", "alpha": 0.45},
                medianprops={"color": "black"},
                showfliers=False,
            )
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(stations, rotation=70, ha="right")
        ax.set_ylabel("Monthly downstream/upstream-gauge ratio")
        ax.set_title(f"Gauge-to-gauge ratio distribution ({upstream_mode})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            [plt.Line2D([0], [0], color=f"C{i}", linewidth=8, alpha=0.45) for i in range(len(sources))],
            sources,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"boxplot_monthly_gauge_to_gauge_ratio_{upstream_mode}.png", dpi=dpi)
        plt.close(fig)


def _plot_gauge_to_gauge_maps(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    background_shape,
) -> None:
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    target_crs = _resolve_plot_crs(("lon", "lat"), None, None)
    plot_background = _prepare_background_shape(background_shape, target_crs)
    background_bounds = _background_bounds(plot_background)
    ratio_norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    diff_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    merged = summary.merge(metadata, on="gauge_id", how="left")
    if {"lon", "lat"}.issubset(merged.columns):
        for (upstream_mode, source), group in merged.groupby(["upstream_mode", "source"], sort=True):
            fig, ax = plt.subplots(figsize=(7, 6))
            _plot_background_shape(ax, plot_background)
            sc = ax.scatter(
                group["lon"],
                group["lat"],
                c=group["downstream_to_upstream_ratio"],
                s=38,
                cmap="RdBu",
                norm=ratio_norm,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.set_title(f"{source} downstream/upstream-gauge ratio ({upstream_mode})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            _apply_map_bounds(ax, background_bounds)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("Ratio (0-2, white=1)")
            fig.tight_layout()
            fig.savefig(map_dir / f"map_{upstream_mode}_{source.lower()}_gauge_to_gauge_ratio.png", dpi=dpi)
            plt.close(fig)

    comp = comparison.merge(metadata, on="gauge_id", how="left")
    if {"lon", "lat"}.issubset(comp.columns):
        for value_col in [col for col in comp.columns if col.endswith("_ratio") and "_minus_" in col]:
            for upstream_mode, group in comp.groupby("upstream_mode", sort=True):
                fig, ax = plt.subplots(figsize=(7, 6))
                _plot_background_shape(ax, plot_background)
                sc = ax.scatter(
                    group["lon"],
                    group["lat"],
                    c=group[value_col],
                    s=38,
                    cmap="RdBu",
                    norm=diff_norm,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_title(f"{value_col.replace('_', ' ')} ({upstream_mode})")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.2)
                _apply_map_bounds(ax, background_bounds)
                cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
                cbar.set_label("Ratio difference (-1 to 1)")
                fig.tight_layout()
                fig.savefig(map_dir / f"map_{upstream_mode}_{value_col}.png", dpi=dpi)
                plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.rapid_file = detect_rapid_file(args.rapid_file, qout_var=args.qout_var)
    print(f"Using RAPID file: {args.rapid_file}")
    output_dir = args.output_dir or args.evaluation_dir / "mass_balance"
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = _resolve_stations(args.stations, args.evaluation_dir, args.period)
    metadata = _read_metadata(args.gauge_metadata, stations)
    background_shape = _read_background_shape(args.background_shapefile)
    graph = _load_graph(args.graph_file)
    source_flat, weights_m2, membership, gauge_nodes = _source_to_gauge_matrix(graph, stations)

    gnn = _read_gnn_timeseries(args.evaluation_dir, args.period, stations)
    global_start = pd.Timestamp(gnn["date"].min()) + pd.Timedelta(days=int(args.trim_days))
    global_end = pd.Timestamp(gnn["date"].max()) - pd.Timedelta(days=int(args.trim_days))
    gnn = gnn[(gnn["date"] >= global_start) & (gnn["date"] <= global_end)].copy()
    if gnn.empty:
        raise ValueError("No GNN rows remain after trim-days filtering.")

    if args.infer_comids_from_nldi:
        comids, mapping = _infer_comids_from_nldi(
            args.rapid_file,
            args.qout_var,
            args.rivid_dim,
            metadata,
            stations,
            args.nldi_comid_cache,
            args.nldi_timeout,
            True,
        )
    elif args.infer_comids_from_rapid_lonlat:
        comids, mapping = _infer_comids_from_rapid_lonlat(args.rapid_file, args.qout_var, args.rivid_dim, metadata, stations)
    else:
        raise ValueError("Use --infer-comids-from-nldi or --infer-comids-from-rapid-lonlat.")

    rapid, rapid_meta = _read_rapid_timeseries(args.rapid_file, stations, comids, args.qout_var, args.rivid_dim, args.time_dim)
    rapid["date"] = pd.to_datetime(rapid["date"]).dt.normalize()
    gnn["date"] = pd.to_datetime(gnn["date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(gnn["date"].unique()))
    runoff = _compute_daily_upstream_runoff(
        forcing_dir=args.forcing_dir,
        runoff_vars=[str(value) for value in args.runoff_vars],
        source_flat=source_flat,
        weights_m2=weights_m2,
        membership=membership,
        stations=stations,
        dates=dates,
        time_dim=args.time_dim_forcing,
        y_dim=args.y_dim,
        x_dim=args.x_dim,
    )

    aligned = gnn.merge(rapid, on=["gauge_id", "date"], how="inner").merge(runoff, on=["gauge_id", "date"], how="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(subset=["gnn", "observed", "rapid", "runoff_m3_day"])
    aligned = _add_volume_columns(aligned)
    if aligned.empty:
        raise ValueError("No overlapping valid GNN/observed/RAPID/runoff rows were found.")

    aligned["year"] = aligned["date"].dt.year
    aligned["month"] = aligned["date"].dt.month
    aligned["window_id"] = ((aligned["date"] - aligned["date"].min()).dt.days // int(args.window_days)).astype(int)
    seasons = _parse_season_windows(args.season_windows)
    aligned["season_window"] = "Other"
    for name, months in seasons.items():
        aligned.loc[aligned["month"].isin(months), "season_window"] = name

    daily_path = output_dir / "daily_mass_balance.csv"
    aligned.to_csv(daily_path, index=False)
    full_summary = _volume_summary(aligned, ["gauge_id"])
    full_summary["gnn_minus_rapid_ratio"] = full_summary["gnn_runoff_ratio"] - full_summary["rapid_runoff_ratio"]
    full_summary.to_csv(output_dir / "mass_balance_by_gauge.csv", index=False)
    by_year = _volume_summary(aligned, ["gauge_id", "year"])
    by_year.to_csv(output_dir / "mass_balance_by_gauge_year.csv", index=False)
    by_season = _volume_summary(aligned[aligned["season_window"] != "Other"], ["gauge_id", "season_window"])
    by_season.to_csv(output_dir / "mass_balance_by_gauge_season.csv", index=False)
    by_window = _volume_summary(aligned, ["gauge_id", "window_id"])
    by_window.to_csv(output_dir / "mass_balance_by_gauge_window.csv", index=False)
    by_month = _monthly_volume_ratios(aligned)
    by_month.to_csv(output_dir / "mass_balance_by_gauge_month.csv", index=False)
    mapping.to_csv(output_dir / "rapid_comid_mapping.csv", index=False)

    _plot_ratio_bars(full_summary, output_dir, args.dpi)
    _plot_scatter(full_summary, output_dir, args.dpi)
    _plot_monthly_ratio_timeseries(by_month, output_dir, args.dpi)
    _plot_cumulative(aligned, output_dir, args.dpi)
    _plot_heatmap(by_year, "gnn_runoff_ratio", output_dir / "heatmap_annual_gnn_runoff_ratio.png", "Annual GNN volume/runoff ratio", args.dpi, 0.0, 2.0)
    _plot_heatmap(by_year, "rapid_runoff_ratio", output_dir / "heatmap_annual_rapid_runoff_ratio.png", "Annual RAPID volume/runoff ratio", args.dpi, 0.0, 2.0)
    _plot_heatmap(by_year.assign(gnn_minus_rapid_ratio=by_year["gnn_runoff_ratio"] - by_year["rapid_runoff_ratio"]), "gnn_minus_rapid_ratio", output_dir / "heatmap_annual_gnn_minus_rapid_ratio.png", "Annual GNN - RAPID volume ratio", args.dpi, -0.5, 0.5)
    _plot_maps(full_summary, metadata, output_dir, args.dpi, background_shape)
    _gauge_to_gauge_diagnostics(aligned, graph, gauge_nodes, metadata, output_dir, args.dpi, background_shape)

    summary = {
        "evaluation_dir": str(args.evaluation_dir),
        "output_dir": str(output_dir),
        "graph_file": str(args.graph_file),
        "forcing_dir": str(args.forcing_dir),
        "rapid": rapid_meta,
        "trim_days": int(args.trim_days),
        "analysis_start": str(aligned["date"].min().date()),
        "analysis_end": str(aligned["date"].max().date()),
        "stations": stations,
        "gauge_nodes": gauge_nodes,
        "daily_rows": int(len(aligned)),
        "tables": [
            "daily_mass_balance.csv",
            "mass_balance_by_gauge.csv",
            "mass_balance_by_gauge_year.csv",
            "mass_balance_by_gauge_season.csv",
            "mass_balance_by_gauge_window.csv",
            "mass_balance_by_gauge_month.csv",
            "gauge_to_gauge_balance/gauge_to_gauge_topology.csv",
            "gauge_to_gauge_balance/gauge_to_gauge_balance_by_gauge.csv",
            "gauge_to_gauge_balance/gauge_to_gauge_balance_by_gauge_month.csv",
            "gauge_to_gauge_balance/gauge_to_gauge_ratio_comparison.csv",
        ],
    }
    (output_dir / "mass_balance_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
