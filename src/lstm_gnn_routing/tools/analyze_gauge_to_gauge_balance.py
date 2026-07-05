"""Gauge-to-gauge streamflow consistency diagnostics.

This tool compares the streamflow at each downstream gauge with the sum of
streamflow at supervised upstream gauges.  It is intentionally different from
the runoff-volume mass-balance tool: the denominator here is streamflow from
upstream gauges, not NoahMP runoff.  That makes the diagnostic useful for asking
whether the GNN systematically amplifies or removes water between gauges, and it
can be applied in the same way to observed, RAPID, and GNN streamflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lstm_gnn_routing.tools.rapid_file import detect_rapid_file

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - geopandas is optional for non-map outputs.
    gpd = None

try:
    import requests
except Exception:  # pragma: no cover - requests is optional with a COMID cache.
    requests = None


M3S_TO_M3DAY = 86400.0
CFS_TO_M3S = 1.0 / 35.314666721489

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


@dataclass(frozen=True)
class GaugeTopology:
    station: str
    node_index: int
    frontier_upstream_stations: tuple[str, ...]
    all_upstream_stations: tuple[str, ...]


def _as_station(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _read_station_list(args: argparse.Namespace) -> list[str]:
    if args.stations:
        if len(args.stations) == 1 and args.stations[0].lower() == "dam_filtered":
            return DAM_FILTERED_GAUGES.copy()
        return [_as_station(s) for s in args.stations]
    if args.basin_file:
        stations: list[str] = []
        with open(args.basin_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    stations.append(_as_station(stripped.split()[0]))
        return stations
    return DAM_FILTERED_GAUGES.copy()


def _read_gauge_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, dtype=str)
    station_col = _find_column(meta.columns, ["site_no", "gage", "gauge", "gauge_id", "station", "station_id"])
    if station_col is None:
        station_col = meta.columns[0]
    meta["station"] = meta[station_col].map(_as_station)

    for col in meta.columns:
        low = col.lower()
        if low in {"lat", "latitude", "dec_lat_va", "site_lat"}:
            meta["lat"] = pd.to_numeric(meta[col], errors="coerce")
        if low in {"lon", "long", "longitude", "dec_long_va", "site_lon"}:
            meta["lon"] = pd.to_numeric(meta[col], errors="coerce")
        if low in {"x", "easting"}:
            meta["x"] = pd.to_numeric(meta[col], errors="coerce")
        if low in {"y", "northing"}:
            meta["y"] = pd.to_numeric(meta[col], errors="coerce")
    return meta


def _metadata_rapid_id_mapping(gauge_meta: pd.DataFrame) -> dict[str, int]:
    """Return station -> RAPID reach id using metadata columns when available.

    Older analysis scripts used an MLID column for RAPID extraction.  Newer
    workflows may call the same concept COMID/rivid.  We accept all of these
    names and prefer metadata over NLDI so RAPID and GNN diagnostics use the
    same gauge-to-reach mapping as the established mass-balance script.
    """
    id_col = _find_column(
        gauge_meta.columns,
        [
            "MLID",
            "mlid",
            "rapid_mlid",
            "rapid_id",
            "rivid",
            "riv_id",
            "COMID",
            "comid",
            "nhdplus_comid",
            "nhd_comid",
        ],
    )
    if id_col is None or "station" not in gauge_meta.columns:
        return {}
    mapping: dict[str, int] = {}
    for _, row in gauge_meta.iterrows():
        station = _as_station(row["station"])
        try:
            mapping[station] = int(float(row[id_col]))
        except Exception:
            continue
    return mapping


def _graph_var(ds: xr.Dataset, candidates: Iterable[str]) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of these variables are present: {', '.join(candidates)}")


def _decode_string_array(values: np.ndarray) -> list[str]:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.dtype.kind in {"S", "U"}:
        decoded = []
        for row in arr:
            text = "".join(x.decode() if isinstance(x, bytes) else str(x) for x in row)
            decoded.append(text.strip().strip("\x00"))
        return decoded
    decoded = []
    for value in arr:
        if isinstance(value, bytes):
            decoded.append(value.decode().strip().strip("\x00"))
        else:
            decoded.append(str(value).strip().strip("\x00"))
    return decoded


def _load_graph_topology(graph_path: Path, stations: list[str]) -> dict[str, GaugeTopology]:
    ds = xr.open_dataset(graph_path)
    try:
        edge_source = _graph_var(ds, ["edge_source", "edge_sources", "source", "src"]).values.astype(int)
        edge_target = _graph_var(ds, ["edge_target", "edge_targets", "target", "dst"]).values.astype(int)
        gauge_node = _graph_var(
            ds,
            [
                "gauge_node_index",
                "gauge_node_indices",
                "gauge_node",
                "gauge_nodes",
                "gauge_target_index",
                "gauge_target_indices",
            ],
        ).values.astype(int)
        raw_ids = _graph_var(ds, ["gauge_id", "gauge_ids", "station_id", "station_ids", "gage_id", "gage_ids"]).values
    except KeyError as exc:
        raise KeyError(
            f"Graph cache {graph_path} is missing expected gauge/topology variable {exc!s}."
        ) from exc
    finally:
        ds.close()

    gauge_ids = [_as_station(x) for x in _decode_string_array(raw_ids)]
    gauge_to_node = dict(zip(gauge_ids, gauge_node))
    selected = [s for s in stations if s in gauge_to_node]
    node_to_gauges: dict[int, list[str]] = defaultdict(list)
    for station in selected:
        node_to_gauges[gauge_to_node[station]].append(station)

    reverse_adj: dict[int, list[int]] = defaultdict(list)
    for src, dst in zip(edge_source, edge_target):
        reverse_adj[int(dst)].append(int(src))

    all_upstream_by_station: dict[str, tuple[str, ...]] = {}
    for station in selected:
        start = int(gauge_to_node[station])
        visited = {start}
        queue: deque[int] = deque([start])
        upstream_nodes: set[int] = set()
        while queue:
            node = queue.popleft()
            for upstream in reverse_adj.get(node, []):
                if upstream in visited:
                    continue
                visited.add(upstream)
                upstream_nodes.add(upstream)
                queue.append(upstream)

        upstream_stations: list[str] = []
        for node in upstream_nodes:
            upstream_stations.extend(node_to_gauges.get(node, []))
        upstream_stations = sorted(s for s in upstream_stations if s != station)
        all_upstream_by_station[station] = tuple(upstream_stations)

    topology: dict[str, GaugeTopology] = {}
    for station in selected:
        all_upstream = set(all_upstream_by_station[station])
        frontier: list[str] = []
        for candidate in sorted(all_upstream):
            # If candidate is upstream of another upstream gauge, summing both would
            # double-count the candidate's water. Keep only the nearest gauge cut-set.
            candidate_is_nested = any(
                candidate in all_upstream_by_station.get(other, ())
                for other in all_upstream
                if other != candidate
            )
            if not candidate_is_nested:
                frontier.append(candidate)
        topology[station] = GaugeTopology(
            station=station,
            node_index=int(gauge_to_node[station]),
            frontier_upstream_stations=tuple(frontier),
            all_upstream_stations=tuple(sorted(all_upstream)),
        )
    return topology


def _read_observed_station(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = _find_column(df.columns, ["date", "datetime", "time"])
    if date_col is None:
        date_col = df.columns[0]
    value_candidates = [
        "streamflow_m3s",
        "flow_m3s",
        "q_m3s",
        "discharge_m3s",
        "00060_Mean",
        "00060",
        "discharge",
        "flow",
        "q",
    ]
    value_col = _find_column(df.columns, value_candidates)
    if value_col is None:
        numeric_cols = [c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if not numeric_cols:
            raise ValueError(f"Could not infer streamflow column in {path}.")
        value_col = numeric_cols[-1]

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.values, index=dates).sort_index()
    series = series[series.index.notna()]
    name = str(value_col).lower()
    if "cfs" in name or "00060" in name:
        series = series * CFS_TO_M3S
    return series.rename(path.stem)


def _load_observed_streamflow(streamflow_dir: Path, stations: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for station in stations:
        candidates = sorted(streamflow_dir.glob(f"*{station}*.csv"))
        if not candidates:
            continue
        data[station] = _read_observed_station(candidates[0])
    if not data:
        raise ValueError(f"No observed streamflow CSVs found in {streamflow_dir}.")
    return pd.DataFrame(data).sort_index()


def _load_gnn_streamflow(evaluation_dir: Path, stations: list[str]) -> pd.DataFrame:
    """Load GNN prediction time series in m3/s from common evaluation outputs."""
    csvs = sorted(evaluation_dir.rglob("*.csv"))
    candidates: list[Path] = []
    for path in csvs:
        name = path.name.lower()
        if any(token in name for token in ["prediction", "timeseries", "streamflow", "forecast"]):
            if "metric" not in name and "history" not in name and "summary" not in name:
                candidates.append(path)

    frames: list[pd.DataFrame] = []
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        date_col = _find_column(df.columns, ["date", "datetime", "time"])
        station_col = _find_column(df.columns, ["station", "gauge", "gauge_id", "site_no", "gage"])
        pred_col = _find_column(
            df.columns,
            [
                "prediction_m3s",
                "pred_m3s",
                "simulated_m3s",
                "gnn_m3s",
                "prediction",
                "pred",
                "simulated",
                "y_pred",
            ],
        )
        if date_col and station_col and pred_col:
            tmp = df[[date_col, station_col, pred_col]].copy()
            tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp["station"] = tmp[station_col].map(_as_station)
            tmp["value"] = pd.to_numeric(tmp[pred_col], errors="coerce")
            wide = tmp.pivot_table(index="date", columns="station", values="value", aggfunc="mean")
            frames.append(wide)
            continue

        # Some evaluation files are already wide: one date column and gauge columns.
        if date_col:
            gauge_cols = [c for c in df.columns if _as_station(c) in stations]
            if gauge_cols:
                wide = df[[date_col, *gauge_cols]].copy()
                wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce")
                wide = wide.set_index(date_col)
                wide = wide.rename(columns={c: _as_station(c) for c in gauge_cols})
                frames.append(wide)

    if not frames:
        found = "\n".join(str(p) for p in candidates[:20])
        raise ValueError(
            f"Could not find a GNN prediction time-series CSV under {evaluation_dir}.\n"
            f"Candidate files checked:\n{found}"
        )
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    keep = [s for s in stations if s in merged.columns]
    if not keep:
        raise ValueError(f"GNN prediction files under {evaluation_dir} did not contain requested gauges.")
    return merged[keep].sort_index()


def _read_comid_cache(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    station_col = _find_column(df.columns, ["station", "site_no", "gauge", "gauge_id"])
    comid_col = _find_column(df.columns, ["comid", "COMID", "nhdplus_comid", "rivid"])
    if station_col is None or comid_col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            out[_as_station(row[station_col])] = int(float(row[comid_col]))
        except Exception:
            continue
    return out


def _write_comid_cache(path: Path | None, mapping: dict[str, int]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"station": k, "comid": v} for k, v in sorted(mapping.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def _get_nldi_comid(station: str) -> int | None:
    if requests is None:
        return None
    url = f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{station}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    features = data.get("features", [])
    if not features:
        return None
    props = features[0].get("properties", {})
    for key in ("comid", "COMID", "nhdplus_comid", "nhd_comid"):
        value = props.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            try:
                return int(float(value))
            except Exception:
                pass
    return None


def _infer_rivid_dim(ds: xr.Dataset, var_name: str) -> str:
    dims = ds[var_name].dims
    for candidate in ("rivid", "COMID", "comid", "feature_id", "feature", "station"):
        if candidate in dims:
            return candidate
    for dim in dims:
        if dim.lower() not in {"time", "date"}:
            return dim
    raise ValueError(f"Could not infer RAPID reach dimension from variable {var_name}.")


def _load_rapid_streamflow(
    rapid_file: Path,
    stations: list[str],
    comid_cache: Path | None,
    infer_comids_from_nldi: bool,
    metadata_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    ds = xr.open_dataset(rapid_file)
    try:
        var_name = "Qout" if "Qout" in ds.data_vars else next(iter(ds.data_vars))
        reach_dim = _infer_rivid_dim(ds, var_name)
        mapping = _read_comid_cache(comid_cache)
        if metadata_mapping:
            mapping.update({k: v for k, v in metadata_mapping.items() if k in stations})
        missing = [s for s in stations if s not in mapping]
        if infer_comids_from_nldi:
            for station in missing:
                comid = _get_nldi_comid(station)
                if comid is not None:
                    mapping[station] = comid
            _write_comid_cache(comid_cache, mapping)

        data: dict[str, pd.Series] = {}
        reach_values = set(np.asarray(ds[reach_dim].values).astype(int).tolist())
        for station in stations:
            comid = mapping.get(station)
            if comid is None or int(comid) not in reach_values:
                continue
            da = ds[var_name].sel({reach_dim: int(comid)})
            frame = da.to_dataframe(name="value").reset_index()
            date_col = _find_column(frame.columns, ["time", "date", "datetime"])
            if date_col is None:
                continue
            series = pd.Series(
                pd.to_numeric(frame["value"], errors="coerce").values,
                index=pd.to_datetime(frame[date_col], errors="coerce"),
            ).sort_index()
            data[station] = series
        if not data:
            raise ValueError(
                f"No requested stations could be extracted from RAPID file {rapid_file}. "
                "Provide --nldi-comid-cache or enable --infer-comids-from-nldi."
            )
        return pd.DataFrame(data).sort_index()
    finally:
        ds.close()


def _to_monthly_volume_m3(streamflow_m3s: pd.DataFrame) -> pd.DataFrame:
    daily_volume = streamflow_m3s * M3S_TO_M3DAY
    return daily_volume.resample("MS").sum(min_count=1)


def _write_period_metadata(
    output_dir: Path,
    observed: pd.DataFrame,
    gnn: pd.DataFrame,
    rapid: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    rows = []
    for source, frame in [("Observed", observed), ("GNN", gnn), ("RAPID", rapid)]:
        rows.append(
            {
                "source": source,
                "start_date": frame.index.min().date().isoformat() if len(frame.index) else "",
                "end_date": frame.index.max().date().isoformat() if len(frame.index) else "",
                "daily_steps": int(len(frame.index)),
                "gauge_columns": int(frame.shape[1]),
            }
        )
    period = {
        "analysis_start_date": start.date().isoformat(),
        "analysis_end_date": end.date().isoformat(),
        "analysis_daily_steps": int((end - start).days + 1),
        "note": (
            "Gauge-to-gauge summary volumes are summed over this full aligned "
            "period. If the GNN evaluation directory only contains test-period "
            "predictions, this aligned period will also be test-only."
        ),
        "source_periods": rows,
    }
    pd.DataFrame(rows).to_csv(output_dir / "analysis_period_by_source.csv", index=False)
    with open(output_dir / "analysis_period.json", "w", encoding="utf-8") as handle:
        json.dump(period, handle, indent=2)


def _compute_gauge_to_gauge(
    monthly_volume: pd.DataFrame,
    topology: dict[str, GaugeTopology],
    source_name: str,
    upstream_mode: str,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for station, topo in sorted(topology.items()):
        if upstream_mode == "all_upstream":
            upstream_candidates = topo.all_upstream_stations
        elif upstream_mode == "frontier":
            upstream_candidates = topo.frontier_upstream_stations
        else:
            raise ValueError(f"Unsupported upstream_mode {upstream_mode!r}.")
        upstream = [s for s in upstream_candidates if s in monthly_volume.columns]
        if not upstream or station not in monthly_volume.columns:
            continue
        downstream = monthly_volume[station]
        upstream_sum = monthly_volume[upstream].sum(axis=1, min_count=1)
        ratio = downstream / upstream_sum.replace(0.0, np.nan)
        incremental = downstream - upstream_sum
        valid = downstream.notna() & upstream_sum.notna() & np.isfinite(ratio)
        if not valid.any():
            continue
        for date in monthly_volume.index[valid]:
            monthly_rows.append(
                {
                    "source": source_name,
                    "upstream_mode": upstream_mode,
                    "station": station,
                    "date": date,
                    "downstream_volume_m3": float(downstream.loc[date]),
                    "upstream_gauge_volume_m3": float(upstream_sum.loc[date]),
                    "downstream_to_upstream_ratio": float(ratio.loc[date]),
                    "incremental_volume_m3": float(incremental.loc[date]),
                    "upstream_gauge_count": len(upstream),
                    "upstream_gauges": ",".join(upstream),
                }
            )
        ds = downstream[valid]
        us = upstream_sum[valid]
        inc = incremental[valid]
        rat = ratio[valid]
        rows.append(
            {
                "source": source_name,
                "upstream_mode": upstream_mode,
                "station": station,
                "upstream_gauge_count": len(upstream),
                "upstream_gauges": ",".join(upstream),
                "months": int(valid.sum()),
                "analysis_start_date": analysis_start.date().isoformat(),
                "analysis_end_date": analysis_end.date().isoformat(),
                "downstream_volume_m3": float(ds.sum()),
                "upstream_gauge_volume_m3": float(us.sum()),
                "downstream_to_upstream_ratio": float(ds.sum() / us.sum()) if us.sum() != 0 else np.nan,
                "median_monthly_ratio": float(rat.median()),
                "mean_monthly_ratio": float(rat.mean()),
                "incremental_volume_m3": float(inc.sum()),
                "incremental_fraction_of_downstream": float(inc.sum() / ds.sum()) if ds.sum() != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(monthly_rows)


def _merge_source_summaries(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    wide = summary.pivot(
        index=["upstream_mode", "station"],
        columns="source",
        values="downstream_to_upstream_ratio",
    )
    for left, right in [("GNN", "Observed"), ("GNN", "RAPID"), ("RAPID", "Observed")]:
        if left in wide.columns and right in wide.columns:
            wide[f"{left}_minus_{right}_ratio"] = wide[left] - wide[right]
    return wide.reset_index()


def _plot_monthly_ratio(monthly: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots" / "monthly_ratio"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if monthly.empty:
        return
    for (mode, station), group in monthly.groupby(["upstream_mode", "station"], sort=True):
        fig, ax = plt.subplots(figsize=(13, 5))
        for source, src_group in group.groupby("source", sort=True):
            src_group = src_group.sort_values("date")
            ax.plot(
                pd.to_datetime(src_group["date"]),
                src_group["downstream_to_upstream_ratio"],
                label=source,
                linewidth=1.8,
            )
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(f"Gauge-to-gauge streamflow ratio | {station} | {mode}")
        ax.set_ylabel("Downstream flow / upstream-gauge flow")
        ax.set_xlabel("Month")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(plot_dir / f"{mode}_{station}_monthly_ratio.png", dpi=220)
        plt.close(fig)


def _plot_incremental(monthly: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots" / "incremental_flow"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if monthly.empty:
        return
    for (mode, station), group in monthly.groupby(["upstream_mode", "station"], sort=True):
        fig, ax = plt.subplots(figsize=(13, 5))
        for source, src_group in group.groupby("source", sort=True):
            src_group = src_group.sort_values("date")
            ax.plot(
                pd.to_datetime(src_group["date"]),
                src_group["incremental_volume_m3"] / 1.0e6,
                label=source,
                linewidth=1.8,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(f"Incremental streamflow between upstream gauges and {station} | {mode}")
        ax.set_ylabel("Monthly incremental volume (million m3)")
        ax.set_xlabel("Month")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(plot_dir / f"{mode}_{station}_monthly_incremental_volume.png", dpi=220)
        plt.close(fig)


def _plot_source_boxplots(monthly: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if monthly.empty:
        return
    for mode, mode_monthly in monthly.groupby("upstream_mode", sort=True):
        stations = sorted(mode_monthly["station"].unique())
        sources = [s for s in ["Observed", "RAPID", "GNN"] if s in set(mode_monthly["source"])]
        positions = np.arange(len(stations))
        width = 0.23
        fig, ax = plt.subplots(figsize=(max(12, len(stations) * 0.45), 6))
        for i, source in enumerate(sources):
            data = [
                mode_monthly[(mode_monthly["station"] == station) & (mode_monthly["source"] == source)][
                    "downstream_to_upstream_ratio"
                ].dropna()
                for station in stations
            ]
            ax.boxplot(
                data,
                positions=positions + (i - (len(sources) - 1) / 2.0) * width,
                widths=width * 0.85,
                patch_artist=True,
                boxprops={"facecolor": f"C{i}", "alpha": 0.45},
                medianprops={"color": "black"},
                showfliers=False,
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(stations, rotation=70, ha="right")
        ax.set_ylabel("Monthly downstream/upstream-gauge ratio")
        ax.set_title(f"Gauge-to-gauge streamflow ratio distribution | {mode}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(
            [plt.Line2D([0], [0], color=f"C{i}", linewidth=8, alpha=0.45) for i in range(len(sources))],
            sources,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"boxplot_monthly_downstream_upstream_ratio_{mode}.png", dpi=220)
        plt.close(fig)


def _plot_maps(
    summary: pd.DataFrame,
    gauge_meta: pd.DataFrame,
    output_dir: Path,
    background_shapefile: Path | None,
) -> None:
    if gpd is None or summary.empty:
        return
    plot_dir = output_dir / "plots" / "maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    coords = gauge_meta[["station", "lat", "lon"]].dropna()
    data = summary.merge(coords, on="station", how="left").dropna(subset=["lat", "lon"])
    if data.empty:
        return

    bg = None
    if background_shapefile and background_shapefile.exists():
        try:
            bg_path = background_shapefile
            if bg_path.is_dir():
                matches: list[Path] = []
                for pattern in ("*.shp", "*.gpkg", "*.geojson", "*.json"):
                    matches.extend(sorted(bg_path.glob(pattern)))
                if matches:
                    bg_path = matches[0]
            bg = gpd.read_file(bg_path)
            if bg.crs is not None:
                bg = bg.to_crs("EPSG:4326")
        except Exception:
            bg = None

    for (mode, source), group in data.groupby(["upstream_mode", "source"], sort=True):
        fig, ax = plt.subplots(figsize=(8, 7))
        if bg is not None and not bg.empty:
            bg.boundary.plot(ax=ax, color="black", linewidth=1.2)
        scatter = ax.scatter(
            group["lon"],
            group["lat"],
            c=group["downstream_to_upstream_ratio"],
            s=34,
            cmap="RdBu_r",
            vmin=0.0,
            vmax=2.0,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Downstream/upstream-gauge ratio")
        ax.set_title(f"{source} gauge-to-gauge ratio | {mode}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if bg is not None and not bg.empty:
            minx, miny, maxx, maxy = bg.total_bounds
            padx = (maxx - minx) * 0.06
            pady = (maxy - miny) * 0.06
            ax.set_xlim(minx - padx, maxx + padx)
            ax.set_ylim(miny - pady, maxy + pady)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(plot_dir / f"map_{mode}_{source.lower()}_downstream_upstream_ratio.png", dpi=240)
        plt.close(fig)

    wide = _merge_source_summaries(summary)
    diff_cols = [c for c in wide.columns if c.endswith("_ratio")]
    diff_data = wide.merge(coords, on="station", how="left").dropna(subset=["lat", "lon"])
    for col in diff_cols:
        if col in {"GNN", "RAPID", "Observed"}:
            continue
        for mode, group in diff_data.groupby("upstream_mode", sort=True):
            fig, ax = plt.subplots(figsize=(8, 7))
            if bg is not None and not bg.empty:
                bg.boundary.plot(ax=ax, color="black", linewidth=1.2)
            scatter = ax.scatter(
                group["lon"],
                group["lat"],
                c=group[col],
                s=34,
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
            cbar.set_label(col.replace("_", " "))
            ax.set_title(f"{col.replace('_', ' ')} | {mode}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if bg is not None and not bg.empty:
                minx, miny, maxx, maxy = bg.total_bounds
                padx = (maxx - minx) * 0.06
                pady = (maxy - miny) * 0.06
                ax.set_xlim(minx - padx, maxx + padx)
                ax.set_ylim(miny - pady, maxy + pady)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.savefig(plot_dir / f"map_{mode}_{col}.png", dpi=240)
            plt.close(fig)


def _write_topology(topology: dict[str, GaugeTopology], output_dir: Path) -> None:
    rows = [
        {
            "station": topo.station,
            "node_index": topo.node_index,
            "frontier_upstream_gauge_count": len(topo.frontier_upstream_stations),
            "frontier_upstream_gauges": ",".join(topo.frontier_upstream_stations),
            "all_upstream_gauge_count": len(topo.all_upstream_stations),
            "all_upstream_gauges": ",".join(topo.all_upstream_stations),
        }
        for topo in sorted(topology.values(), key=lambda x: x.station)
    ]
    pd.DataFrame(rows).to_csv(output_dir / "gauge_upstream_topology.csv", index=False)
    with open(output_dir / "gauge_upstream_topology.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def _ensure_gnn_evaluation(args: argparse.Namespace, default_output_dir: Path) -> Path:
    """Return a GNN evaluation directory, creating it from --run-dir if needed."""
    if args.evaluation_dir is not None:
        return args.evaluation_dir
    if args.run_dir is None:
        raise ValueError("Provide either --evaluation-dir or --run-dir.")

    eval_dir = args.gnn_evaluation_output_dir or (args.run_dir / "evaluation_full_period_best_final_stage_model")
    marker_files = [
        eval_dir / "predictions.csv",
        eval_dir / "streamflow_timeseries.csv",
        eval_dir / "evaluation_timeseries.csv",
        eval_dir / "metrics_summary.csv",
    ]
    has_existing_csv = eval_dir.exists() and any(eval_dir.rglob("*.csv"))
    if has_existing_csv and not args.force_evaluate:
        print(f"Using existing GNN evaluation directory: {eval_dir}")
        return eval_dir

    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "lstm_gnn_routing.cli.main",
        "evaluate",
        "--run-dir",
        str(args.run_dir),
        "--period",
        args.evaluation_period,
        "--output-dir",
        str(eval_dir),
    ]
    if args.checkpoint_file:
        cmd.extend(["--checkpoint-file", str(args.checkpoint_file)])
    if args.noah_config:
        cmd.extend(["--noah-config", str(args.noah_config)])

    print("Running GNN evaluation before gauge-to-gauge analysis:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    if not any(eval_dir.rglob("*.csv")):
        raise ValueError(f"GNN evaluation completed but no CSV files were found in {eval_dir}.")
    return eval_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=None, help="Existing GNN evaluation output directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory. If provided without --evaluation-dir, the tool first runs GNN evaluation.")
    parser.add_argument("--checkpoint-file", type=Path, default=None, help="Checkpoint to evaluate when --run-dir is used.")
    parser.add_argument("--noah-config", type=Path, default=None, help="Optional evaluation config override for full-period/continuous inference.")
    parser.add_argument("--evaluation-period", default="test", choices=["validation", "test"], help="Dataset period passed to the evaluator. Use --noah-config to redefine this period as full/continuous.")
    parser.add_argument("--gnn-evaluation-output-dir", type=Path, default=None, help="Where to save newly generated GNN evaluation outputs.")
    parser.add_argument("--force-evaluate", action="store_true", help="Re-run GNN evaluation even if output CSVs already exist.")
    parser.add_argument("--graph", type=Path, required=True, help="Routing graph NetCDF cache.")
    parser.add_argument("--streamflow-dir", type=Path, default=Path("data/streamflow/daily"))
    parser.add_argument("--gauge-metadata", type=Path, default=Path("data/streamflow/30_gauges_IN_LAMBERT.csv"))
    parser.add_argument("--basin-file", type=Path, default=None)
    parser.add_argument("--rapid-file", type=Path, default=None)
    parser.add_argument("--nldi-comid-cache", type=Path, default=Path("data/streamflow/nldi_comid_cache.csv"))
    parser.add_argument("--infer-comids-from-nldi", action="store_true")
    parser.add_argument("--background-shapefile", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stations", nargs="*", default=["dam_filtered"])
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--expected-min-gnn-days",
        type=int,
        default=0,
        help=(
            "Fail if the loaded GNN prediction series has fewer than this many daily "
            "steps. Useful for preventing accidental test-only mass-balance analysis."
        ),
    )
    args = parser.parse_args()

    provisional_output = args.output_dir or Path("gauge_to_gauge_balance")
    evaluation_dir = _ensure_gnn_evaluation(args, provisional_output)
    output_dir = args.output_dir or (evaluation_dir / "gauge_to_gauge_balance")
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = _read_station_list(args)
    args.rapid_file = detect_rapid_file(args.rapid_file)
    print(f"Using RAPID file: {args.rapid_file}")
    gauge_meta = _read_gauge_metadata(args.gauge_metadata)
    topology = _load_graph_topology(args.graph, stations)
    _write_topology(topology, output_dir)

    stations_with_topology = sorted(topology)
    observed = _load_observed_streamflow(args.streamflow_dir, stations_with_topology)
    gnn = _load_gnn_streamflow(evaluation_dir, stations_with_topology)
    rapid = _load_rapid_streamflow(
        args.rapid_file,
        stations_with_topology,
        args.nldi_comid_cache,
        args.infer_comids_from_nldi,
        _metadata_rapid_id_mapping(gauge_meta),
    )

    start = pd.to_datetime(args.start_date) if args.start_date else max(
        df.index.min() for df in [observed, gnn, rapid] if not df.empty
    )
    end = pd.to_datetime(args.end_date) if args.end_date else min(
        df.index.max() for df in [observed, gnn, rapid] if not df.empty
    )
    observed = observed.loc[start:end]
    gnn = gnn.loc[start:end]
    rapid = rapid.loc[start:end]

    if args.expected_min_gnn_days and len(gnn.index) < args.expected_min_gnn_days:
        raise ValueError(
            f"GNN prediction period has only {len(gnn.index)} daily steps after alignment, "
            f"but --expected-min-gnn-days={args.expected_min_gnn_days}. "
            "Use a continuous/full-period evaluation directory or lower the threshold."
        )
    if "evaluation_test" in str(evaluation_dir).lower() and not args.start_date and not args.end_date:
        print(
            "WARNING: evaluation directory name contains 'evaluation_test'. "
            "The whole-period summary will still sum all loaded GNN dates, but if this "
            "folder is test-only then the analysis remains test-only. Use a continuous "
            "evaluation directory for full train+validation+test consistency."
        )

    _write_period_metadata(output_dir, observed, gnn, rapid, start, end)

    monthly_sources = {
        "Observed": _to_monthly_volume_m3(observed),
        "GNN": _to_monthly_volume_m3(gnn),
        "RAPID": _to_monthly_volume_m3(rapid),
    }
    summaries: list[pd.DataFrame] = []
    monthly_tables: list[pd.DataFrame] = []
    for source, monthly in monthly_sources.items():
        for upstream_mode in ("frontier", "all_upstream"):
            summary, monthly_table = _compute_gauge_to_gauge(
                monthly,
                topology,
                source,
                upstream_mode,
                start,
                end,
            )
            summaries.append(summary)
            monthly_tables.append(monthly_table)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    monthly = pd.concat(monthly_tables, ignore_index=True) if monthly_tables else pd.DataFrame()
    comparison = _merge_source_summaries(summary)

    summary.to_csv(output_dir / "gauge_to_gauge_balance_summary.csv", index=False)
    monthly.to_csv(output_dir / "gauge_to_gauge_balance_monthly.csv", index=False)
    comparison.to_csv(output_dir / "gauge_to_gauge_balance_ratio_comparison.csv", index=False)

    _plot_monthly_ratio(monthly, output_dir)
    _plot_incremental(monthly, output_dir)
    _plot_source_boxplots(monthly, output_dir)
    _plot_maps(summary, gauge_meta, output_dir, args.background_shapefile)

    print(f"Saved gauge-to-gauge balance outputs to {output_dir}")
    print(f"Analyzed {summary['station'].nunique() if not summary.empty else 0} downstream gauges with upstream gauges.")


if __name__ == "__main__":
    main()
