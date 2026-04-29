from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_GAUGES = [
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

SEASON_DEFINITIONS = {
    "Dec-May": [12, 1, 2, 3, 4, 5],
    "Jul-Sep": [7, 8, 9],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create diagnostic plots/tables for evaluated routing-model streamflow "
            "predictions."
        )
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help="Directory containing test_timeseries.csv and test_metrics_by_gauge.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Used to auto-detect an evaluation_test_* directory when --evaluation-dir is omitted.",
    )
    parser.add_argument(
        "--checkpoint-stem",
        default="best_final_stage_model",
        help="Evaluation directory suffix to use with --run-dir, e.g. best_final_stage_model or best_model.",
    )
    parser.add_argument("--period", default="test", help="Evaluation period label used in file names.")
    parser.add_argument(
        "--gauge-metadata",
        type=Path,
        default=Path("data/streamflow/30_gauges_IN_LAMBERT.csv"),
        help="CSV with basin_id, lat, lon, x, y columns.",
    )
    parser.add_argument(
        "--background-shapefile",
        type=Path,
        default=Path("data/HUC4/Salt_and_Verde.shp"),
        help="Optional basin/HUC shapefile to draw behind spatial maps.",
    )
    parser.add_argument(
        "--map-dem",
        type=Path,
        default=Path("data/DEM/basin_srtm_dem_conditioned_on_forcing_grid.nc"),
        help="DEM/grid NetCDF used to infer the CRS for projected gauge x/y map coordinates.",
    )
    parser.add_argument(
        "--gauge-crs",
        default=None,
        help="Optional explicit CRS for gauge x/y map coordinates, e.g. EPSG:5070 or a WKT/proj string.",
    )
    parser.add_argument(
        "--gauges",
        nargs="*",
        default=DEFAULT_GAUGES,
        help="Gauge IDs to include. Defaults to the dam-filtered training gauges.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <evaluation-dir>/analysis.",
    )
    parser.add_argument(
        "--kgess-benchmark",
        type=float,
        default=1.0 - math.sqrt(2.0),
        help="Benchmark KGE used for KGESS=(KGE-benchmark)/(1-benchmark). Default is mean-flow benchmark 1-sqrt(2).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution for saved figures.",
    )
    parser.add_argument(
        "--max-gauge-plots",
        type=int,
        default=0,
        help="Limit per-gauge plot count for quick tests. Use 0 for all gauges.",
    )
    return parser.parse_args()


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def _resolve_evaluation_dir(args: argparse.Namespace) -> Path:
    if args.evaluation_dir is not None:
        return args.evaluation_dir
    if args.run_dir is None:
        raise ValueError("Provide either --evaluation-dir or --run-dir.")
    candidate = args.run_dir / f"evaluation_{args.period}_{args.checkpoint_stem}"
    if candidate.is_dir():
        return candidate
    matches = sorted(args.run_dir.glob(f"evaluation_{args.period}_*"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No evaluation_{args.period}_* directory found under {args.run_dir}")
    options = "\n".join(str(path) for path in matches)
    raise ValueError(
        f"Multiple evaluation directories found under {args.run_dir}. "
        f"Use --evaluation-dir explicitly.\n{options}"
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def _clean_metric_name(value: str) -> str:
    return value.replace("daily_", "").replace("window_", "")


def _read_inputs(evaluation_dir: Path, period: str, gauges: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timeseries_path = evaluation_dir / f"{period}_timeseries.csv"
    metrics_path = evaluation_dir / f"{period}_metrics_by_gauge.csv"
    summary_path = evaluation_dir / f"{period}_metrics_summary.json"
    if not timeseries_path.is_file():
        raise FileNotFoundError(f"Missing timeseries file: {timeseries_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    gauge_set = {str(value) for value in gauges}
    ts = pd.read_csv(timeseries_path, dtype={"gauge_id": str}, parse_dates=["date"])
    ts = ts[ts["gauge_id"].isin(gauge_set)].copy()
    ts = ts.sort_values(["gauge_id", "date"]).reset_index(drop=True)
    if ts.empty:
        raise ValueError("No timeseries rows remained after gauge filtering.")

    metrics = pd.read_csv(metrics_path, dtype={"gauge_id": str})
    metrics = metrics[metrics["gauge_id"].isin(gauge_set)].copy()
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    return ts, metrics, summary


def _read_metadata(path: Path, gauges: Iterable[str]) -> pd.DataFrame:
    gauge_set = {str(value) for value in gauges}
    if not path.is_file():
        return pd.DataFrame({"gauge_id": sorted(gauge_set)})
    meta = pd.read_csv(path, dtype={"basin_id": str, "gauge_id": str})
    if "gauge_id" not in meta and "basin_id" in meta:
        meta = meta.rename(columns={"basin_id": "gauge_id"})
    meta["gauge_id"] = meta["gauge_id"].astype(str)
    return meta[meta["gauge_id"].isin(gauge_set)].copy()


def _read_background_shape(path: Path | None):
    if path is None or not path.is_file():
        return None
    try:
        import geopandas as gpd
    except Exception:
        return _read_background_shape_fallback(path)
    try:
        return gpd.read_file(path)
    except Exception:
        return _read_background_shape_fallback(path)


def _read_background_shape_fallback(path: Path):
    try:
        import struct

        segments = []
        with path.open("rb") as fp:
            fp.seek(100)
            while True:
                rec_header = fp.read(8)
                if len(rec_header) < 8:
                    break
                _, content_len_words = struct.unpack(">2i", rec_header)
                content = fp.read(content_len_words * 2)
                if len(content) < 44:
                    continue
                shape_type = struct.unpack("<i", content[:4])[0]
                if shape_type not in (5, 15, 25):
                    continue
                num_parts, num_points = struct.unpack("<2i", content[36:44])
                parts_offset = 44
                points_offset = parts_offset + 4 * num_parts
                parts = list(struct.unpack("<" + "i" * num_parts, content[parts_offset:points_offset]))
                points = []
                for index in range(num_points):
                    start = points_offset + 16 * index
                    points.append(struct.unpack("<2d", content[start : start + 16]))
                parts.append(num_points)
                for index in range(num_parts):
                    segment = points[parts[index] : parts[index + 1]]
                    if segment:
                        segments.append(segment)
        if not segments:
            return None
        prj_text = path.with_suffix(".prj").read_text(errors="ignore") if path.with_suffix(".prj").is_file() else ""
        return {
            "segments": segments,
            "is_geographic": "GEOGCS" in prj_text.upper() or "GEOGCRS" in prj_text.upper(),
        }
    except Exception:
        return None


def _infer_dem_crs(path: Path | None):
    if path is None or not path.is_file():
        return None
    try:
        import xarray as xr
        from pyproj import CRS
    except Exception:
        return None
    try:
        ds = xr.open_dataset(path)
        try:
            for coord_name in ("spatial_ref", "crs"):
                if coord_name not in ds:
                    continue
                attrs = ds[coord_name].attrs
                wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
                if wkt:
                    return CRS.from_wkt(str(wkt))
            return None
        finally:
            ds.close()
    except Exception:
        return None


def _resolve_plot_crs(coord_cols: tuple[str, str], map_dem: Path | None, gauge_crs: str | None):
    try:
        from pyproj import CRS
    except Exception:
        return None
    if coord_cols == ("lon", "lat"):
        return CRS.from_epsg(4269)
    if gauge_crs:
        try:
            return CRS.from_user_input(gauge_crs)
        except Exception:
            return None
    return _infer_dem_crs(map_dem)


def _prepare_background_shape(background_shape, target_crs):
    if isinstance(background_shape, dict):
        return background_shape
    if background_shape is None or target_crs is None:
        return background_shape
    try:
        if background_shape.crs is None:
            return background_shape
        return background_shape.to_crs(target_crs)
    except Exception:
        return background_shape


def _background_is_geographic(background_shape) -> bool:
    if isinstance(background_shape, dict):
        return bool(background_shape.get("is_geographic", False))
    try:
        return bool(background_shape is not None and background_shape.crs is not None and background_shape.crs.is_geographic)
    except Exception:
        return False


def _select_map_coord_cols(metric_frame: pd.DataFrame, background_shape) -> tuple[str, str] | None:
    # The Salt-Verde HUC4 shapefile is geographic. Prefer lon/lat when available so
    # the basin boundary and gauges are guaranteed to share the same map CRS.
    if _background_is_geographic(background_shape) and {"lon", "lat"}.issubset(metric_frame.columns):
        return ("lon", "lat")
    if {"x", "y"}.issubset(metric_frame.columns):
        return ("x", "y")
    if {"lon", "lat"}.issubset(metric_frame.columns):
        return ("lon", "lat")
    return None


def _plot_background_shape(ax, background_shape) -> None:
    if background_shape is None:
        return
    try:
        if isinstance(background_shape, dict):
            for segment in background_shape.get("segments", []):
                xs = [point[0] for point in segment]
                ys = [point[1] for point in segment]
                ax.fill(xs, ys, facecolor="#f2efe8", edgecolor="#344e41", linewidth=1.0, alpha=0.55, zorder=0)
                ax.plot(xs, ys, color="#344e41", linewidth=1.25, zorder=1)
            return
        if getattr(background_shape, "empty", True):
            return
        background_shape.plot(
            ax=ax,
            facecolor="#f2efe8",
            edgecolor="#5f6f52",
            linewidth=1.15,
            alpha=0.55,
            zorder=0,
        )
        background_shape.boundary.plot(ax=ax, color="#344e41", linewidth=1.35, zorder=1)
    except Exception:
        return


def _kge_components(pred: np.ndarray, obs: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(pred) & np.isfinite(obs)
    count = int(valid.sum())
    if count < 2:
        return {
            "valid_points": count,
            "kge": np.nan,
            "nse": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
        }
    p = pred[valid].astype(float)
    o = obs[valid].astype(float)
    eps = np.finfo(float).eps
    mean_p = float(np.mean(p))
    mean_o = float(np.mean(o))
    std_p = float(np.std(p))
    std_o = float(np.std(o))
    corr = float(np.corrcoef(p, o)[0, 1]) if std_p > eps and std_o > eps else np.nan
    alpha = std_p / (std_o + eps)
    beta = (mean_p + eps) / (mean_o + eps)
    kge = 1.0 - math.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite(corr) else np.nan
    error = p - o
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    denom = float(np.sum((o - mean_o) ** 2))
    nse = 1.0 - float(np.sum(error**2)) / denom if denom > eps else np.nan
    return {
        "valid_points": count,
        "kge": kge,
        "nse": nse,
        "mse": mse,
        "rmse": rmse,
        "correlation": corr,
        "alpha": alpha,
        "beta": beta,
    }


def _metric_rows_by_period(ts: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groupby_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in ts.groupby(groupby_key, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(_kge_components(group["prediction"].to_numpy(), group["observation"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def _add_time_groups(ts: pd.DataFrame) -> pd.DataFrame:
    result = ts.copy()
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["month_name"] = result["date"].dt.strftime("%b")
    result["water_year"] = result["date"].dt.year + (result["month"] >= 10).astype(int)
    for season_name, months in SEASON_DEFINITIONS.items():
        result[f"season_{_safe_name(season_name)}"] = result["month"].isin(months)
    return result


def _write_tables(
    *,
    ts: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    kgess_benchmark: float,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _add_time_groups(ts)

    daily_metrics = _metric_rows_by_period(ts, ["gauge_id"])
    daily_metrics["daily_kgess"] = (daily_metrics["kge"] - kgess_benchmark) / (1.0 - kgess_benchmark)
    daily_metrics = daily_metrics.rename(
        columns={
            "kge": "daily_kge_recomputed",
            "nse": "daily_nse_recomputed",
            "mse": "daily_mse_recomputed",
            "rmse": "daily_rmse_recomputed",
        }
    )
    merged_metrics = metrics.merge(daily_metrics, on="gauge_id", how="outer")
    if "daily_kge" in merged_metrics:
        merged_metrics["daily_kgess"] = (merged_metrics["daily_kge"] - kgess_benchmark) / (1.0 - kgess_benchmark)
    elif "daily_kge_recomputed" in merged_metrics:
        merged_metrics["daily_kgess"] = merged_metrics["daily_kgess"]
    merged_metrics = merged_metrics.merge(metadata, on="gauge_id", how="left")
    merged_metrics.to_csv(output_dir / "metrics_by_gauge_with_metadata.csv", index=False)

    monthly_metrics = _metric_rows_by_period(ts, ["gauge_id", "month", "month_name"])
    monthly_metrics["kgess"] = (monthly_metrics["kge"] - kgess_benchmark) / (1.0 - kgess_benchmark)
    monthly_metrics.to_csv(output_dir / "monthly_metrics_by_gauge.csv", index=False)

    seasonal_frames = []
    for season_name in SEASON_DEFINITIONS:
        col = f"season_{_safe_name(season_name)}"
        seasonal = ts[ts[col]].copy()
        seasonal["season"] = season_name
        seasonal_frames.append(seasonal)
    seasonal_ts = pd.concat(seasonal_frames, ignore_index=True)
    seasonal_metrics = _metric_rows_by_period(seasonal_ts, ["gauge_id", "season"])
    seasonal_metrics["kgess"] = (seasonal_metrics["kge"] - kgess_benchmark) / (1.0 - kgess_benchmark)
    seasonal_metrics.to_csv(output_dir / "seasonal_metrics_by_gauge.csv", index=False)

    flow_summary = (
        ts.groupby("gauge_id")[["prediction", "observation"]]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    flow_summary.columns = ["_".join(str(part) for part in col if str(part)) for col in flow_summary.columns]
    flow_summary.to_csv(output_dir / "streamflow_distribution_summary.csv", index=False)

    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {
        "timeseries": ts,
        "metrics": merged_metrics,
        "monthly_metrics": monthly_metrics,
        "seasonal_metrics": seasonal_metrics,
        "seasonal_timeseries": seasonal_ts,
    }


def _plot_timeseries(plt, gauge_ts: pd.DataFrame, gauge_id: str, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(gauge_ts["date"], gauge_ts["observation"], color="#1b4d89", linewidth=1.2, label="Observed")
    ax.plot(gauge_ts["date"], gauge_ts["prediction"], color="#c23b22", linewidth=1.0, alpha=0.85, label="Predicted")
    ax.set_title(f"{gauge_id} daily streamflow")
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{gauge_id}_timeseries.png", dpi=dpi)
    plt.close(fig)


def _plot_scatter(plt, gauge_ts: pd.DataFrame, gauge_id: str, output_dir: Path, dpi: int) -> None:
    valid = gauge_ts[["prediction", "observation"]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return
    max_value = float(valid.max().max())
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(valid["observation"], valid["prediction"], s=8, alpha=0.35, color="#355c7d", edgecolors="none")
    ax.plot([0, max_value], [0, max_value], color="#c23b22", linewidth=1.0)
    ax.set_title(f"{gauge_id} predicted vs observed")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{gauge_id}_scatter.png", dpi=dpi)
    plt.close(fig)


def _plot_flow_duration(plt, gauge_ts: pd.DataFrame, gauge_id: str, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for column, label, color in [
        ("observation", "Observed", "#1b4d89"),
        ("prediction", "Predicted", "#c23b22"),
    ]:
        values = np.sort(gauge_ts[column].dropna().to_numpy(dtype=float))[::-1]
        if values.size == 0:
            continue
        exceedance = 100.0 * (np.arange(1, values.size + 1) / (values.size + 1))
        ax.plot(exceedance, values, color=color, linewidth=1.4, label=label)
    ax.set_title(f"{gauge_id} flow duration curve")
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Streamflow")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{gauge_id}_flow_duration.png", dpi=dpi)
    plt.close(fig)


def _paired_boxplot(
    plt,
    frame: pd.DataFrame,
    *,
    group_col: str,
    group_order: list[Any],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for idx, group_value in enumerate(group_order, start=1):
        subset = frame[frame[group_col] == group_value]
        obs = subset["observation"].dropna().to_numpy(dtype=float)
        pred = subset["prediction"].dropna().to_numpy(dtype=float)
        data.extend([obs, pred])
        positions.extend([idx - 0.18, idx + 0.18])
        colors.extend(["#1b4d89", "#c23b22"])
        labels.append(str(group_value))
    if not any(values.size for values in data):
        return
    fig_width = max(8.0, 0.65 * len(group_order))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    box = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.0)
    ax.set_xticks(range(1, len(group_order) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Streamflow")
    ax.grid(True, axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color="#1b4d89", linewidth=6, alpha=0.6, label="Observed"),
        plt.Line2D([0], [0], color="#c23b22", linewidth=6, alpha=0.6, label="Predicted"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_metric_heatmap(
    plt,
    metric_frame: pd.DataFrame,
    *,
    value_col: str,
    column_col: str,
    column_order: list[Any],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    pivot = metric_frame.pivot(index="gauge_id", columns=column_col, values=value_col).reindex(columns=column_order)
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(column_order)), max(6.0, 0.26 * len(pivot))))
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, aspect="auto", cmap="RdYlBu", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels([str(value) for value in column_order], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label(_clean_metric_name(value_col).upper())
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_map(
    plt,
    metric_frame: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    output_path: Path,
    dpi: int,
    line2d,
    background_shape,
    map_dem: Path | None,
    gauge_crs: str | None,
) -> None:
    coord_cols = _select_map_coord_cols(metric_frame, background_shape)
    if coord_cols is None or metric_col not in metric_frame.columns:
        return
    x_col, y_col = coord_cols
    data = metric_frame.dropna(subset=[x_col, y_col, metric_col]).copy()
    if data.empty:
        return
    values = data[metric_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    if metric_col.lower().endswith("kge") or "kgess" in metric_col.lower() or metric_col.lower().endswith("nse"):
        vmin, vmax, cmap = 0.0, 1.0, "YlGnBu"
        values = np.clip(values, vmin, vmax)
    else:
        vmin, vmax, cmap = None, None, "viridis"
    target_crs = _resolve_plot_crs(coord_cols, map_dem, gauge_crs)
    plot_background = _prepare_background_shape(background_shape, target_crs)
    _plot_background_shape(ax, plot_background)
    sc = ax.scatter(
        data[x_col],
        data[y_col],
        c=values,
        s=56,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude" if coord_cols == ("lon", "lat") else "Lambert x")
    ax.set_ylabel("Latitude" if coord_cols == ("lon", "lat") else "Lambert y")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label(f"{metric_col} clipped to [0, 1]" if vmin == 0.0 and vmax == 1.0 else metric_col)
    if coord_cols == ("lon", "lat"):
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_summary_bars(plt, metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    metric_cols = [col for col in ["daily_kge", "daily_kgess", "daily_nse", "daily_rmse"] if col in metrics.columns]
    for metric_col in metric_cols:
        frame = metrics[["gauge_id", metric_col]].dropna().sort_values(metric_col)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.28 * len(frame))))
        colors = np.where(frame[metric_col].to_numpy(dtype=float) >= 0.0, "#4c956c", "#c23b22")
        ax.barh(frame["gauge_id"], frame[metric_col], color=colors, alpha=0.78)
        ax.axvline(0.0, color="#222222", linewidth=0.9)
        ax.set_title(f"Gauge {metric_col}")
        ax.set_xlabel(metric_col)
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"gauge_bar_{metric_col}.png", dpi=dpi)
        plt.close(fig)


def _make_plots(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
    max_gauge_plots: int,
    background_shape,
    map_dem: Path | None,
    gauge_crs: str | None,
) -> None:
    plt, Line2D = _require_matplotlib()
    plot_dir = output_dir / "plots"
    gauge_dir = plot_dir / "gauges"
    map_dir = plot_dir / "maps"
    heatmap_dir = plot_dir / "heatmaps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gauge_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    ts = tables["timeseries"]
    metrics = tables["metrics"]
    monthly_metrics = tables["monthly_metrics"]
    seasonal_metrics = tables["seasonal_metrics"]
    seasonal_ts = tables["seasonal_timeseries"]

    _plot_summary_bars(plt, metrics, plot_dir, dpi)

    for metric_col in ["daily_kge", "daily_kgess", "daily_nse", "daily_rmse"]:
        if metric_col in metrics.columns:
            _plot_map(
                plt,
                metrics,
                metric_col=metric_col,
                title=f"Spatial map of {metric_col}",
                output_path=map_dir / f"map_{metric_col}.png",
                dpi=dpi,
                line2d=Line2D,
                background_shape=background_shape,
                map_dem=map_dem,
                gauge_crs=gauge_crs,
            )

    month_order = list(range(1, 13))
    season_order = list(SEASON_DEFINITIONS.keys())
    for value_col in ["kge", "kgess", "nse", "rmse"]:
        if value_col in monthly_metrics:
            _plot_metric_heatmap(
                plt,
                monthly_metrics,
                value_col=value_col,
                column_col="month",
                column_order=month_order,
                title=f"Monthly {value_col.upper()} by gauge",
                output_path=heatmap_dir / f"monthly_{value_col}_heatmap.png",
                dpi=dpi,
            )
        if value_col in seasonal_metrics:
            _plot_metric_heatmap(
                plt,
                seasonal_metrics,
                value_col=value_col,
                column_col="season",
                column_order=season_order,
                title=f"Seasonal {value_col.upper()} by gauge",
                output_path=heatmap_dir / f"seasonal_{value_col}_heatmap.png",
                dpi=dpi,
            )

    gauge_ids = sorted(ts["gauge_id"].unique().tolist())
    if max_gauge_plots and max_gauge_plots > 0:
        gauge_ids = gauge_ids[:max_gauge_plots]
    for gauge_id in gauge_ids:
        gauge_ts = ts[ts["gauge_id"] == gauge_id].copy()
        gauge_output = gauge_dir / gauge_id
        gauge_output.mkdir(parents=True, exist_ok=True)
        _plot_timeseries(plt, gauge_ts, gauge_id, gauge_output, dpi)
        _plot_scatter(plt, gauge_ts, gauge_id, gauge_output, dpi)
        _plot_flow_duration(plt, gauge_ts, gauge_id, gauge_output, dpi)
        _paired_boxplot(
            plt,
            gauge_ts,
            group_col="month",
            group_order=month_order,
            title=f"{gauge_id} monthly streamflow distribution",
            output_path=gauge_output / f"{gauge_id}_monthly_boxplot.png",
            dpi=dpi,
        )
        gauge_season_ts = seasonal_ts[seasonal_ts["gauge_id"] == gauge_id].copy()
        _paired_boxplot(
            plt,
            gauge_season_ts,
            group_col="season",
            group_order=season_order,
            title=f"{gauge_id} seasonal streamflow distribution",
            output_path=gauge_output / f"{gauge_id}_seasonal_boxplot.png",
            dpi=dpi,
        )


def main() -> None:
    args = _parse_args()
    evaluation_dir = _resolve_evaluation_dir(args)
    output_dir = args.output_dir or (evaluation_dir / "analysis")
    gauges = [str(value) for value in args.gauges]
    ts, metrics, summary = _read_inputs(evaluation_dir, args.period, gauges)
    metadata = _read_metadata(args.gauge_metadata, gauges)
    background_shape = _read_background_shape(args.background_shapefile)
    tables = _write_tables(
        ts=ts,
        metrics=metrics,
        metadata=metadata,
        summary=summary,
        output_dir=output_dir,
        kgess_benchmark=float(args.kgess_benchmark),
    )
    _make_plots(
        tables,
        output_dir=output_dir,
        dpi=int(args.dpi),
        max_gauge_plots=int(args.max_gauge_plots),
        background_shape=background_shape,
        map_dem=args.map_dem,
        gauge_crs=args.gauge_crs,
    )
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
