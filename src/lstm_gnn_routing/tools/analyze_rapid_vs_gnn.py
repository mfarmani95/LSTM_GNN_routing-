from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_STATIONS = ["09508500", "09499000", "09498500"]
DEFAULT_COMIDS = [20438416, 22431630, 22442058]
DEFAULT_DAM_FILTERED_STATIONS = [
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
DEFAULT_EVALUATION_DIR = Path(
    "runs/After_transfer_scaling_center_mapping_derived_hydrology_stage5_dam_filtered/"
    "evaluation_test_best_final_stage_model"
)
DEFAULT_RAPID_FILE = Path("/xdisk/tyferre/farmani/Graph_Routing/RAPID_13nd.nc")
KGESS_BENCHMARK = 1.0 - math.sqrt(2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RAPID Qout against the best GNN-routing evaluation and observed "
            "USGS streamflow for selected gauges."
        )
    )
    parser.add_argument("--rapid-file", type=Path, default=DEFAULT_RAPID_FILE)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--period", default="test")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=DEFAULT_STATIONS,
        help=(
            "Gauge ids to compare. Use 'dam_filtered' for the 23 supervised "
            "dam-filtered gauges, or combine with --stations-from-evaluation "
            "to read gauges from the evaluation timeseries."
        ),
    )
    parser.add_argument("--comids", nargs="+", type=int, default=DEFAULT_COMIDS)
    parser.add_argument(
        "--stations-from-evaluation",
        action="store_true",
        help="Use all gauge ids present in <evaluation-dir>/<period>_timeseries.csv.",
    )
    parser.add_argument(
        "--infer-comids-from-rapid-lonlat",
        action="store_true",
        help=(
            "Infer nearest RAPID rivid/COMID from RAPID WGS84 lon/lat coordinates "
            "and gauge metadata lon/lat. Lambert x/y columns are not used for RAPID selection."
        ),
    )
    parser.add_argument(
        "--infer-comids-from-nldi",
        action="store_true",
        help=(
            "Resolve each USGS gauge to an NHDPlus COMID through the USGS NLDI API, "
            "then extract RAPID Qout at that COMID before computing metrics."
        ),
    )
    parser.add_argument(
        "--nldi-comid-cache",
        type=Path,
        default=Path("data/streamflow/usgs_nldi_comids.csv"),
        help="CSV cache for USGS gauge to NLDI COMID lookups.",
    )
    parser.add_argument("--nldi-timeout", type=float, default=15.0)
    parser.add_argument(
        "--no-nldi-lonlat-fallback",
        action="store_true",
        help="Fail instead of falling back to nearest RAPID WGS84 lon/lat when NLDI lookup or RAPID COMID matching fails.",
    )
    parser.add_argument("--qout-var", default="Qout")
    parser.add_argument("--rivid-dim", default=None, help="RAPID river id dimension/coordinate. Auto-detected by default.")
    parser.add_argument("--time-dim", default=None, help="RAPID time coordinate. Auto-detected by default.")
    parser.add_argument(
        "--gauge-metadata",
        type=Path,
        default=Path("data/streamflow/30_gauges_IN_LAMBERT.csv"),
        help="Gauge metadata CSV with basin_id/gauge_id and x/y or lon/lat columns.",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <evaluation-dir>/rapid_comparison.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--monthly-aggregation",
        choices=["sum", "mean"],
        default="sum",
        help="Aggregation used for monthly/seasonal boxplots. Default follows requested monthly sums.",
    )
    return parser.parse_args()


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _kge_components(pred: np.ndarray, obs: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(pred) & np.isfinite(obs)
    count = int(valid.sum())
    if count < 2:
        return {
            "valid_points": count,
            "kge": np.nan,
            "kgess": np.nan,
            "nse": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "pbias": np.nan,
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
    kgess = (kge - KGESS_BENCHMARK) / (1.0 - KGESS_BENCHMARK) if np.isfinite(kge) else np.nan
    error = p - o
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    denom = float(np.sum((o - mean_o) ** 2))
    nse = 1.0 - float(np.sum(error**2)) / denom if denom > eps else np.nan
    bias = float(np.mean(error))
    pbias = 100.0 * float(np.sum(error)) / (float(np.sum(o)) + eps)
    return {
        "valid_points": count,
        "kge": kge,
        "kgess": kgess,
        "nse": nse,
        "mse": mse,
        "rmse": rmse,
        "bias": bias,
        "pbias": pbias,
        "correlation": corr,
        "alpha": alpha,
        "beta": beta,
    }


def _read_gnn_timeseries(evaluation_dir: Path, period: str, stations: list[str]) -> pd.DataFrame:
    path = evaluation_dir / f"{period}_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GNN evaluation timeseries: {path}")
    station_set = {str(value) for value in stations}
    ts = pd.read_csv(path, dtype={"gauge_id": str}, parse_dates=["date"])
    ts = ts[ts["gauge_id"].isin(station_set)].copy()
    if ts.empty:
        raise ValueError(f"No requested stations were found in {path}")
    return ts.rename(columns={"prediction": "gnn", "observation": "observed"})[
        ["gauge_id", "date", "gnn", "observed"]
    ]


def _read_evaluation_stations(evaluation_dir: Path, period: str) -> list[str]:
    path = evaluation_dir / f"{period}_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GNN evaluation timeseries: {path}")
    ts = pd.read_csv(path, dtype={"gauge_id": str}, usecols=["gauge_id"])
    stations = sorted(ts["gauge_id"].dropna().astype(str).unique().tolist())
    if not stations:
        raise ValueError(f"No gauge_id values were found in {path}")
    return stations


def _infer_rivid_name(qout: xr.DataArray, requested: str | None) -> str:
    if requested:
        return requested
    candidates = ["rivid", "COMID", "comid", "reach", "feature_id", "river_id"]
    for name in candidates:
        if name in qout.coords or name in qout.dims:
            return name
    non_time_dims = [dim for dim in qout.dims if "time" not in dim.lower()]
    if len(non_time_dims) == 1:
        return non_time_dims[0]
    raise ValueError(f"Could not infer RAPID river id dimension from Qout dims={qout.dims}")


def _infer_time_name(qout: xr.DataArray, requested: str | None) -> str:
    if requested:
        return requested
    for name in ["time", "Time", "datetime", "date"]:
        if name in qout.coords or name in qout.dims:
            return name
    time_like = [dim for dim in qout.dims if "time" in dim.lower()]
    if len(time_like) == 1:
        return time_like[0]
    raise ValueError(f"Could not infer RAPID time coordinate from Qout dims={qout.dims}")


def _select_rapid_series(qout: xr.DataArray, rivid_name: str, time_name: str, comid: int) -> pd.Series:
    if rivid_name in qout.coords:
        selected = qout.sel({rivid_name: comid})
    else:
        values = qout[rivid_name].values if rivid_name in qout else None
        if values is None:
            raise ValueError(f"RAPID Qout has no coordinate or variable named {rivid_name!r}.")
        matches = np.where(np.asarray(values) == comid)[0]
        if matches.size == 0:
            raise KeyError(f"COMID {comid} was not found in RAPID river ids.")
        selected = qout.isel({rivid_name: int(matches[0])})
    frame = selected.to_dataframe(name="rapid").reset_index()
    frame["date"] = pd.to_datetime(frame[time_name])
    return frame.set_index("date")["rapid"].astype(float).sort_index()


def _read_rapid_timeseries(
    rapid_file: Path,
    stations: list[str],
    comids: list[int],
    qout_var: str,
    rivid_dim: str | None,
    time_dim: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(stations) != len(comids):
        raise ValueError("--stations and --comids must have the same length.")
    if not rapid_file.is_file():
        raise FileNotFoundError(f"Missing RAPID file: {rapid_file}")
    ds = xr.open_dataset(rapid_file)
    if qout_var not in ds:
        raise KeyError(f"Variable {qout_var!r} not found in {rapid_file}. Variables: {list(ds.data_vars)}")
    qout = ds[qout_var]
    rivid_name = _infer_rivid_name(qout, rivid_dim)
    time_name = _infer_time_name(qout, time_dim)
    units = str(qout.attrs.get("units", "")).strip()
    rows = []
    for station, comid in zip(stations, comids):
        series = _select_rapid_series(qout, rivid_name, time_name, int(comid))
        frame = series.rename("rapid").reset_index()
        frame["gauge_id"] = str(station)
        frame["comid"] = int(comid)
        rows.append(frame[["gauge_id", "comid", "date", "rapid"]])
    metadata = {
        "rapid_file": str(rapid_file),
        "qout_variable": qout_var,
        "qout_units": units or "unknown",
        "rivid_name": rivid_name,
        "time_name": time_name,
        "qout_dims": list(qout.dims),
        "qout_shape": [int(value) for value in qout.shape],
    }
    return pd.concat(rows, ignore_index=True), metadata


def _align_daily(gnn: pd.DataFrame, rapid: pd.DataFrame) -> pd.DataFrame:
    gnn = gnn.copy()
    rapid = rapid.copy()
    gnn["date"] = pd.to_datetime(gnn["date"]).dt.floor("D")
    rapid["date"] = pd.to_datetime(rapid["date"]).dt.floor("D")
    rapid = rapid.groupby(["gauge_id", "comid", "date"], as_index=False)["rapid"].mean()
    aligned = gnn.merge(rapid, on=["gauge_id", "date"], how="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.dropna(subset=["observed", "gnn", "rapid"])
    return aligned.sort_values(["gauge_id", "date"]).reset_index(drop=True)


def _monthly_aggregate(aligned: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    monthly = aligned.copy()
    monthly["month_start"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    monthly["year"] = monthly["month_start"].dt.year
    monthly["month"] = monthly["month_start"].dt.month
    monthly["month_name"] = monthly["month_start"].dt.strftime("%b")
    group_cols = ["gauge_id", "comid", "month_start", "year", "month", "month_name"]
    if aggregation == "sum":
        result = monthly.groupby(group_cols, as_index=False)[["observed", "gnn", "rapid"]].sum()
    else:
        result = monthly.groupby(group_cols, as_index=False)[["observed", "gnn", "rapid"]].mean()
    result["season_window"] = np.select(
        [result["month"].isin([12, 1, 2, 3, 4, 5]), result["month"].isin([6, 7, 8, 9])],
        ["Dec-May", "Jun-Sep"],
        default="Other",
    )
    return result


def _metric_rows(data: pd.DataFrame, *, date_col: str, value_cols: list[str], group_cols: list[str], scale: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for model_col in value_cols:
            row = dict(zip(group_cols, keys))
            row["model"] = "GNN" if model_col == "gnn" else "RAPID"
            row["scale"] = scale
            row["start_date"] = group[date_col].min()
            row["end_date"] = group[date_col].max()
            row.update(_kge_components(group[model_col].to_numpy(), group["observed"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def _read_metadata(path: Path, stations: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame({"gauge_id": stations})
    meta = pd.read_csv(path, dtype={"basin_id": str, "gauge_id": str})
    if "gauge_id" not in meta and "basin_id" in meta:
        meta = meta.rename(columns={"basin_id": "gauge_id"})
    meta["gauge_id"] = meta["gauge_id"].astype(str)
    return meta[meta["gauge_id"].isin(set(stations))].copy()


def _load_nldi_cache(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path, dtype={"gauge_id": str})
    except Exception:
        return {}
    if not {"gauge_id", "comid"}.issubset(frame.columns):
        return {}
    result: dict[str, int] = {}
    for _, row in frame.dropna(subset=["gauge_id", "comid"]).iterrows():
        try:
            result[str(row["gauge_id"])] = int(row["comid"])
        except Exception:
            continue
    return result


def _write_nldi_cache(path: Path, cache: dict[str, int]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"gauge_id": gauge_id, "comid": int(comid)} for gauge_id, comid in sorted(cache.items())]
    )
    frame.to_csv(path, index=False)


def _fetch_nldi_comid(site_no: str, timeout: float) -> int | None:
    import urllib.error
    import urllib.request

    url = f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site_no}"
    request = urllib.request.Request(url, headers={"User-Agent": "lstm-gnn-routing-analysis/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    features = data.get("features", [])
    if not features:
        return None
    props = features[0].get("properties", {})
    for key in ("comid", "COMID", "nhdplus_comid"):
        value = props.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _nearest_rapid_index(
    rapid_lon: np.ndarray,
    rapid_lat: np.ndarray,
    valid_reaches: np.ndarray,
    gauge_lon: float,
    gauge_lat: float,
) -> tuple[int, float]:
    dx = (rapid_lon - gauge_lon) * math.cos(math.radians(gauge_lat))
    dy = rapid_lat - gauge_lat
    dist2 = dx * dx + dy * dy
    dist2[~valid_reaches] = np.inf
    nearest_index = int(np.argmin(dist2))
    distance_km = math.sqrt(float(dist2[nearest_index])) * 111.32
    return nearest_index, distance_km


def _infer_comids_from_rapid_lonlat(
    rapid_file: Path,
    qout_var: str,
    rivid_dim: str | None,
    gauge_metadata: pd.DataFrame,
    stations: list[str],
) -> tuple[list[int], pd.DataFrame]:
    if not rapid_file.is_file():
        raise FileNotFoundError(f"Missing RAPID file: {rapid_file}")
    if not {"gauge_id", "lon", "lat"}.issubset(gauge_metadata.columns):
        raise ValueError(
            "COMID inference needs gauge metadata columns gauge_id, lon, and lat. "
            f"Available columns: {list(gauge_metadata.columns)}"
        )

    ds = xr.open_dataset(rapid_file)
    try:
        if qout_var not in ds:
            raise KeyError(f"Variable {qout_var!r} not found in {rapid_file}. Variables: {list(ds.data_vars)}")
        qout = ds[qout_var]
        rivid_name = _infer_rivid_name(qout, rivid_dim)
        if "lon" not in ds or "lat" not in ds:
            raise KeyError("RAPID file must contain lon and lat coordinates/variables to infer nearest COMIDs.")

        rapid_lon = np.asarray(ds["lon"].values, dtype=float)
        rapid_lat = np.asarray(ds["lat"].values, dtype=float)
        if rivid_name in ds.coords:
            rivid_values = np.asarray(ds[rivid_name].values)
        elif rivid_name in ds:
            rivid_values = np.asarray(ds[rivid_name].values)
        else:
            raise KeyError(f"Could not find RAPID river ids named {rivid_name!r}.")

        valid_reaches = np.isfinite(rapid_lon) & np.isfinite(rapid_lat)
        if valid_reaches.sum() == 0:
            raise ValueError("RAPID lon/lat arrays do not contain any finite reach coordinates.")

        metadata_by_station = gauge_metadata.drop_duplicates("gauge_id").set_index("gauge_id")
        comids: list[int] = []
        rows: list[dict[str, Any]] = []
        for station in stations:
            if station not in metadata_by_station.index:
                raise KeyError(f"Gauge {station} is missing from gauge metadata, so RAPID COMID cannot be inferred.")
            gauge_lon = float(metadata_by_station.at[station, "lon"])
            gauge_lat = float(metadata_by_station.at[station, "lat"])
            if not (math.isfinite(gauge_lon) and math.isfinite(gauge_lat)):
                raise ValueError(f"Gauge {station} has non-finite lon/lat metadata.")

            # RAPID reach coordinates are WGS84. Always infer COMID from gauge
            # lon/lat here; projected Lambert x/y columns are intentionally unused.
            nearest_index, distance_km = _nearest_rapid_index(
                rapid_lon,
                rapid_lat,
                valid_reaches,
                gauge_lon,
                gauge_lat,
            )
            comid = int(rivid_values[nearest_index])
            comids.append(comid)
            rows.append(
                {
                    "gauge_id": station,
                    "comid": comid,
                    "mapping_method": "nearest_rapid_wgs84_lonlat",
                    "gauge_lon": gauge_lon,
                    "gauge_lat": gauge_lat,
                    "rapid_lon": float(rapid_lon[nearest_index]),
                    "rapid_lat": float(rapid_lat[nearest_index]),
                    "nearest_distance_km": distance_km,
                }
            )
    finally:
        ds.close()

    return comids, pd.DataFrame(rows)


def _infer_comids_from_nldi(
    rapid_file: Path,
    qout_var: str,
    rivid_dim: str | None,
    gauge_metadata: pd.DataFrame,
    stations: list[str],
    cache_path: Path,
    timeout: float,
    fallback_to_lonlat: bool,
) -> tuple[list[int], pd.DataFrame]:
    if not rapid_file.is_file():
        raise FileNotFoundError(f"Missing RAPID file: {rapid_file}")
    if not {"gauge_id", "lon", "lat"}.issubset(gauge_metadata.columns):
        raise ValueError(
            "NLDI COMID mapping still needs gauge metadata columns gauge_id, lon, and lat "
            "so fallback distance/provenance can be reported."
        )

    cache = _load_nldi_cache(cache_path)
    cache_changed = False
    ds = xr.open_dataset(rapid_file)
    try:
        if qout_var not in ds:
            raise KeyError(f"Variable {qout_var!r} not found in {rapid_file}. Variables: {list(ds.data_vars)}")
        qout = ds[qout_var]
        rivid_name = _infer_rivid_name(qout, rivid_dim)
        if "lon" not in ds or "lat" not in ds:
            raise KeyError("RAPID file must contain lon and lat coordinates/variables for mapping provenance.")

        rapid_lon = np.asarray(ds["lon"].values, dtype=float)
        rapid_lat = np.asarray(ds["lat"].values, dtype=float)
        if rivid_name in ds.coords:
            rivid_values = np.asarray(ds[rivid_name].values)
        elif rivid_name in ds:
            rivid_values = np.asarray(ds[rivid_name].values)
        else:
            raise KeyError(f"Could not find RAPID river ids named {rivid_name!r}.")
        rivid_lookup = {int(value): index for index, value in enumerate(rivid_values)}

        valid_reaches = np.isfinite(rapid_lon) & np.isfinite(rapid_lat)
        metadata_by_station = gauge_metadata.drop_duplicates("gauge_id").set_index("gauge_id")
        comids: list[int] = []
        rows: list[dict[str, Any]] = []
        for station in stations:
            if station not in metadata_by_station.index:
                raise KeyError(f"Gauge {station} is missing from gauge metadata, so RAPID COMID cannot be mapped.")
            gauge_lon = float(metadata_by_station.at[station, "lon"])
            gauge_lat = float(metadata_by_station.at[station, "lat"])
            if not (math.isfinite(gauge_lon) and math.isfinite(gauge_lat)):
                raise ValueError(f"Gauge {station} has non-finite lon/lat metadata.")

            nldi_comid = cache.get(station)
            if nldi_comid is None:
                nldi_comid = _fetch_nldi_comid(station, timeout)
                if nldi_comid is not None:
                    cache[station] = int(nldi_comid)
                    cache_changed = True

            mapping_method = "nldi_usgs_api"
            if nldi_comid is not None and int(nldi_comid) in rivid_lookup:
                comid = int(nldi_comid)
                rapid_index = int(rivid_lookup[comid])
                dx = (rapid_lon[rapid_index] - gauge_lon) * math.cos(math.radians(gauge_lat))
                dy = rapid_lat[rapid_index] - gauge_lat
                distance_km = math.sqrt(float(dx * dx + dy * dy)) * 111.32
            elif fallback_to_lonlat:
                rapid_index, distance_km = _nearest_rapid_index(
                    rapid_lon,
                    rapid_lat,
                    valid_reaches,
                    gauge_lon,
                    gauge_lat,
                )
                comid = int(rivid_values[rapid_index])
                mapping_method = (
                    "nldi_missing_fallback_nearest_rapid_wgs84_lonlat"
                    if nldi_comid is None
                    else "nldi_comid_not_in_rapid_fallback_nearest_rapid_wgs84_lonlat"
                )
            else:
                raise KeyError(
                    f"NLDI COMID for gauge {station} was {nldi_comid}, "
                    "but it was missing or not present in the RAPID rivid coordinate."
                )

            comids.append(comid)
            rows.append(
                {
                    "gauge_id": station,
                    "comid": comid,
                    "nldi_comid": nldi_comid,
                    "mapping_method": mapping_method,
                    "gauge_lon": gauge_lon,
                    "gauge_lat": gauge_lat,
                    "rapid_lon": float(rapid_lon[rapid_index]),
                    "rapid_lat": float(rapid_lat[rapid_index]),
                    "nearest_distance_km": distance_km,
                }
            )
    finally:
        ds.close()
        if cache_changed:
            _write_nldi_cache(cache_path, cache)

    return comids, pd.DataFrame(rows)


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


def _select_map_coord_cols(frame: pd.DataFrame, background_shape) -> tuple[str, str] | None:
    # The Salt-Verde HUC4 shapefile is geographic. Prefer lon/lat when available so
    # the basin boundary and gauges are guaranteed to share the same map CRS.
    if _background_is_geographic(background_shape) and {"lon", "lat"}.issubset(frame.columns):
        return ("lon", "lat")
    if {"x", "y"}.issubset(frame.columns):
        return ("x", "y")
    if {"lon", "lat"}.issubset(frame.columns):
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


def _background_bounds(background_shape) -> tuple[float, float, float, float] | None:
    if background_shape is None:
        return None
    try:
        if isinstance(background_shape, dict):
            points = [point for segment in background_shape.get("segments", []) for point in segment]
            if not points:
                return None
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        if getattr(background_shape, "empty", True):
            return None
        minx, miny, maxx, maxy = background_shape.total_bounds
        return (float(minx), float(miny), float(maxx), float(maxy))
    except Exception:
        return None


def _apply_map_bounds(ax, bounds: tuple[float, float, float, float] | None, pad_fraction: float = 0.04) -> None:
    if bounds is None:
        return
    minx, miny, maxx, maxy = bounds
    if not all(math.isfinite(value) for value in bounds):
        return
    x_span = max(maxx - minx, 1.0e-9)
    y_span = max(maxy - miny, 1.0e-9)
    ax.set_xlim(minx - pad_fraction * x_span, maxx + pad_fraction * x_span)
    ax.set_ylim(miny - pad_fraction * y_span, maxy + pad_fraction * y_span)


def _write_tables(
    output_dir: Path,
    aligned_daily: pd.DataFrame,
    monthly: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    rapid_metadata: dict[str, Any],
    rapid_mapping: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_daily.to_csv(output_dir / "rapid_gnn_observed_daily_aligned.csv", index=False)
    monthly.to_csv(output_dir / "rapid_gnn_observed_monthly.csv", index=False)
    metrics_with_meta = metrics.merge(metadata, on="gauge_id", how="left")
    if rapid_mapping is not None and not rapid_mapping.empty:
        mapping_cols = [
            "gauge_id",
            "comid",
            "nldi_comid",
            "mapping_method",
            "gauge_lon",
            "gauge_lat",
            "rapid_lon",
            "rapid_lat",
            "nearest_distance_km",
        ]
        metrics_with_meta = metrics_with_meta.merge(
            rapid_mapping[[col for col in mapping_cols if col in rapid_mapping.columns]],
            on=["gauge_id", "comid"],
            how="left",
        )
    metrics_with_meta.to_csv(output_dir / "rapid_gnn_metrics.csv", index=False)
    summary = {
        "rapid": rapid_metadata,
        "stations": sorted(aligned_daily["gauge_id"].unique().tolist()),
        "daily_start": str(aligned_daily["date"].min().date()) if not aligned_daily.empty else None,
        "daily_end": str(aligned_daily["date"].max().date()) if not aligned_daily.empty else None,
        "note": (
            "RAPID Qout is assumed to be m3/s; no unit conversion is applied. "
            "When --infer-comids-from-rapid-lonlat is used, RAPID KGE/KGESS are computed "
            "only after selecting each station COMID from WGS84 gauge lon/lat and RAPID reach lon/lat."
        ),
    }
    (output_dir / "rapid_gnn_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def _plot_daily_timeseries(plt, data: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gauges = list(data["gauge_id"].drop_duplicates())
    fig, axes = plt.subplots(len(gauges), 1, figsize=(15, 4.2 * len(gauges)), sharex=True)
    if len(gauges) == 1:
        axes = [axes]
    for ax, gauge_id in zip(axes, gauges):
        subset = data[data["gauge_id"] == gauge_id]
        ax.plot(subset["date"], subset["observed"], color="#1f4e79", linewidth=1.3, label="Observed")
        ax.plot(subset["date"], subset["gnn"], color="#c44900", linewidth=1.0, alpha=0.85, label="GNN")
        ax.plot(subset["date"], subset["rapid"], color="#2a9d8f", linewidth=1.0, alpha=0.85, label="RAPID")
        ax.set_title(f"{gauge_id} daily streamflow")
        ax.set_ylabel("Streamflow (m3/s)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_timeseries_rapid_gnn_observed.png", dpi=dpi)
    plt.close(fig)


def _plot_monthly_timeseries(plt, monthly: pd.DataFrame, output_dir: Path, dpi: int, aggregation: str) -> None:
    gauges = list(monthly["gauge_id"].drop_duplicates())
    ylabel = "Monthly sum of daily streamflow (m3/s)" if aggregation == "sum" else "Monthly mean streamflow (m3/s)"
    fig, axes = plt.subplots(len(gauges), 1, figsize=(15, 4.2 * len(gauges)), sharex=True)
    if len(gauges) == 1:
        axes = [axes]
    for ax, gauge_id in zip(axes, gauges):
        subset = monthly[monthly["gauge_id"] == gauge_id]
        ax.plot(subset["month_start"], subset["observed"], color="#1f4e79", marker="o", markersize=2.5, label="Observed")
        ax.plot(subset["month_start"], subset["gnn"], color="#c44900", marker="o", markersize=2.5, label="GNN")
        ax.plot(subset["month_start"], subset["rapid"], color="#2a9d8f", marker="o", markersize=2.5, label="RAPID")
        ax.set_title(f"{gauge_id} monthly {aggregation}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(output_dir / f"monthly_{aggregation}_timeseries_rapid_gnn_observed.png", dpi=dpi)
    plt.close(fig)


def _plot_season_boxplots(plt, monthly: pd.DataFrame, output_dir: Path, dpi: int, aggregation: str) -> None:
    plot_data = monthly[monthly["season_window"].isin(["Dec-May", "Jun-Sep"])].copy()
    long = plot_data.melt(
        id_vars=["gauge_id", "month_start", "season_window"],
        value_vars=["observed", "gnn", "rapid"],
        var_name="source",
        value_name="streamflow",
    )
    source_order = ["observed", "gnn", "rapid"]
    colors = {"observed": "#1f4e79", "gnn": "#c44900", "rapid": "#2a9d8f"}
    ylabel = "Monthly sum of daily streamflow (m3/s)" if aggregation == "sum" else "Monthly mean streamflow (m3/s)"
    gauges = list(long["gauge_id"].drop_duplicates())
    fig, axes = plt.subplots(len(gauges), 1, figsize=(10.5, 4.0 * len(gauges)), sharey=False)
    if len(gauges) == 1:
        axes = [axes]
    for ax, gauge_id in zip(axes, gauges):
        subset = long[long["gauge_id"] == gauge_id]
        positions = []
        labels = []
        data = []
        box_colors = []
        for season_index, season in enumerate(["Dec-May", "Jun-Sep"]):
            base = season_index * 4
            for source_index, source in enumerate(source_order):
                values = subset[(subset["season_window"] == season) & (subset["source"] == source)]["streamflow"].dropna()
                positions.append(base + source_index + 1)
                labels.append(f"{season}\n{source.upper() if source == 'gnn' else source.title()}")
                data.append(values.to_numpy())
                box_colors.append(colors[source])
        bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(f"{gauge_id} seasonal monthly {aggregation} distribution")
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"seasonal_dec_may_jun_sep_monthly_{aggregation}_boxplots.png", dpi=dpi)
    plt.close(fig)


def _plot_metric_bars(plt, metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    daily = metrics[metrics["scale"] == "daily"].copy()
    if daily.empty:
        return
    gauges = list(daily["gauge_id"].drop_duplicates())
    metric_names = ["kgess", "kge", "nse", "rmse", "pbias"]
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(11.5, 3.1 * len(metric_names)), sharex=True)
    x = np.arange(len(gauges))
    width = 0.36
    for ax, metric in zip(axes, metric_names):
        gnn_values = []
        rapid_values = []
        for gauge_id in gauges:
            gauge_rows = daily[daily["gauge_id"] == gauge_id]
            gnn_values.append(_safe_float(gauge_rows[gauge_rows["model"] == "GNN"][metric].iloc[0]))
            rapid_values.append(_safe_float(gauge_rows[gauge_rows["model"] == "RAPID"][metric].iloc[0]))
        ax.bar(x - width / 2, gnn_values, width, label="GNN", color="#c44900", alpha=0.85)
        ax.bar(x + width / 2, rapid_values, width, label="RAPID", color="#2a9d8f", alpha=0.85)
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.25)
        if metric in {"kgess", "kge", "nse"}:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(gauges)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_metric_bars_rapid_vs_gnn.png", dpi=dpi)
    plt.close(fig)


def _plot_scatter(plt, data: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gauges = list(data["gauge_id"].drop_duplicates())
    fig, axes = plt.subplots(1, len(gauges), figsize=(5.0 * len(gauges), 4.7), sharex=False, sharey=False)
    if len(gauges) == 1:
        axes = [axes]
    for ax, gauge_id in zip(axes, gauges):
        subset = data[data["gauge_id"] == gauge_id]
        max_value = float(subset[["observed", "gnn", "rapid"]].max().max())
        ax.scatter(subset["observed"], subset["gnn"], s=10, alpha=0.35, label="GNN", color="#c44900")
        ax.scatter(subset["observed"], subset["rapid"], s=10, alpha=0.35, label="RAPID", color="#2a9d8f")
        ax.plot([0, max_value], [0, max_value], color="black", linewidth=0.9)
        ax.set_title(gauge_id)
        ax.set_xlabel("Observed (m3/s)")
        ax.set_ylabel("Simulation (m3/s)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_scatter_rapid_gnn_vs_observed.png", dpi=dpi)
    plt.close(fig)


def _plot_spatial_metric_comparison(
    plt,
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    background_shape,
    map_dem: Path | None,
    gauge_crs: str | None,
    *,
    metric: str,
    output_name: str,
) -> None:
    daily = metrics[metrics["scale"] == "daily"].merge(metadata, on="gauge_id", how="left")
    if daily.empty or metric not in daily.columns:
        return
    coord_cols = _select_map_coord_cols(daily, background_shape)
    if coord_cols is None:
        return
    x_col, y_col = coord_cols
    daily = daily.dropna(subset=[x_col, y_col, metric]).copy()
    if daily.empty:
        return
    model_order = ["GNN", "RAPID"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.88, wspace=0.06)
    target_crs = _resolve_plot_crs(coord_cols, map_dem, gauge_crs)
    plot_background = _prepare_background_shape(background_shape, target_crs)
    background_bounds = _background_bounds(plot_background)
    scatter = None
    for ax, model in zip(axes, model_order):
        group = daily[daily["model"] == model].copy()
        _plot_background_shape(ax, plot_background)
        if group.empty:
            ax.set_visible(False)
            continue
        values = group[metric].clip(lower=0.0, upper=1.0)
        sc = ax.scatter(
            group[x_col],
            group[y_col],
            c=values,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            s=70,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        scatter = sc
        ax.set_title(model)
        ax.set_xlabel("Longitude" if coord_cols == ("lon", "lat") else coord_cols[0])
        ax.grid(True, alpha=0.25)
        if coord_cols == ("lon", "lat"):
            ax.set_aspect("equal", adjustable="box")
        _apply_map_bounds(ax, background_bounds)
    axes[0].set_ylabel("Latitude" if coord_cols == ("lon", "lat") else coord_cols[1])
    fig.suptitle(f"Daily {metric.upper()} spatial comparison: GNN vs RAPID")
    if scatter is not None:
        cax = fig.add_axes([0.91, 0.16, 0.022, 0.66])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(f"{metric.upper()} clipped to [0, 1]")
    fig.savefig(output_dir / output_name, dpi=dpi)
    plt.close(fig)


def _plot_spatial_kgess_difference(
    plt,
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    background_shape,
    map_dem: Path | None,
    gauge_crs: str | None,
) -> None:
    daily = metrics[(metrics["scale"] == "daily") & (metrics["model"].isin(["GNN", "RAPID"]))].copy()
    if daily.empty or "kgess" not in daily.columns:
        return
    pivot = daily.pivot_table(index=["gauge_id", "comid"], columns="model", values="kgess", aggfunc="first").reset_index()
    if not {"GNN", "RAPID"}.issubset(pivot.columns):
        return
    pivot["kgess_difference"] = pivot["GNN"] - pivot["RAPID"]
    plot_data = pivot.merge(metadata, on="gauge_id", how="left")
    coord_cols = _select_map_coord_cols(plot_data, background_shape)
    if coord_cols is None:
        return
    x_col, y_col = coord_cols
    plot_data = plot_data.dropna(subset=[x_col, y_col, "kgess_difference"]).copy()
    if plot_data.empty:
        return

    value_limit = 0.5
    target_crs = _resolve_plot_crs(coord_cols, map_dem, gauge_crs)
    plot_background = _prepare_background_shape(background_shape, target_crs)
    background_bounds = _background_bounds(plot_background)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.10, top=0.90)
    _plot_background_shape(ax, plot_background)
    sc = ax.scatter(
        plot_data[x_col],
        plot_data[y_col],
        c=plot_data["kgess_difference"],
        cmap="RdBu",
        vmin=-value_limit,
        vmax=value_limit,
        s=70,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(alpha=0.0)
    ax.set_title("Daily KGESS Difference: GNN - RAPID")
    ax.set_xlabel("Longitude" if coord_cols == ("lon", "lat") else coord_cols[0])
    ax.set_ylabel("Latitude" if coord_cols == ("lon", "lat") else coord_cols[1])
    ax.grid(True, alpha=0.25)
    if coord_cols == ("lon", "lat"):
        ax.set_aspect("equal", adjustable="box")
    _apply_map_bounds(ax, background_bounds)
    cax = fig.add_axes([0.88, 0.17, 0.026, 0.66])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("KGESS(GNN) - KGESS(RAPID)")
    fig.savefig(output_dir / "map_daily_kgess_difference_gnn_minus_rapid.png", dpi=dpi)
    plt.close(fig)


def _write_plots(
    output_dir: Path,
    aligned_daily: pd.DataFrame,
    monthly: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    dpi: int,
    aggregation: str,
    background_shape,
    map_dem: Path | None,
    gauge_crs: str | None,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _require_matplotlib()
    _plot_daily_timeseries(plt, aligned_daily, plot_dir, dpi)
    _plot_monthly_timeseries(plt, monthly, plot_dir, dpi, aggregation)
    _plot_season_boxplots(plt, monthly, plot_dir, dpi, aggregation)
    _plot_metric_bars(plt, metrics, plot_dir, dpi)
    _plot_scatter(plt, aligned_daily, plot_dir, dpi)
    for metric, output_name in [
        ("kge", "map_daily_kge_gnn_vs_rapid.png"),
        ("kgess", "map_daily_kgess_gnn_vs_rapid.png"),
        ("kgess", "map_daily_kgess_clipped_0_1.png"),
    ]:
        _plot_spatial_metric_comparison(
            plt,
            metrics,
            metadata,
            plot_dir,
            dpi,
            background_shape,
            map_dem,
            gauge_crs,
            metric=metric,
            output_name=output_name,
        )
    _plot_spatial_kgess_difference(
        plt,
        metrics,
        metadata,
        plot_dir,
        dpi,
        background_shape,
        map_dem,
        gauge_crs,
    )


def main() -> None:
    args = _parse_args()
    if args.stations_from_evaluation:
        stations = _read_evaluation_stations(args.evaluation_dir, args.period)
    elif len(args.stations) == 1 and str(args.stations[0]).lower() in {"dam_filtered", "dam-filtered", "trained"}:
        stations = DEFAULT_DAM_FILTERED_STATIONS.copy()
    else:
        stations = [str(value) for value in args.stations]
    output_dir = args.output_dir or args.evaluation_dir / "rapid_comparison"

    metadata = _read_metadata(args.gauge_metadata, stations)
    inferred_mapping = None
    if args.infer_comids_from_nldi:
        comids, inferred_mapping = _infer_comids_from_nldi(
            args.rapid_file,
            args.qout_var,
            args.rivid_dim,
            metadata,
            stations,
            args.nldi_comid_cache,
            args.nldi_timeout,
            fallback_to_lonlat=not args.no_nldi_lonlat_fallback,
        )
    elif args.infer_comids_from_rapid_lonlat:
        comids, inferred_mapping = _infer_comids_from_rapid_lonlat(
            args.rapid_file,
            args.qout_var,
            args.rivid_dim,
            metadata,
            stations,
        )
    else:
        comids = [int(value) for value in args.comids]

    gnn = _read_gnn_timeseries(args.evaluation_dir, args.period, stations)
    rapid, rapid_metadata = _read_rapid_timeseries(
        args.rapid_file,
        stations,
        comids,
        args.qout_var,
        args.rivid_dim,
        args.time_dim,
    )
    aligned_daily = _align_daily(gnn, rapid)
    if aligned_daily.empty:
        raise ValueError("No overlapping valid daily values were found between GNN, observed, and RAPID.")
    monthly = _monthly_aggregate(aligned_daily, args.monthly_aggregation)

    daily_metrics = _metric_rows(
        aligned_daily,
        date_col="date",
        value_cols=["gnn", "rapid"],
        group_cols=["gauge_id", "comid"],
        scale="daily",
    )
    monthly_metrics = _metric_rows(
        monthly,
        date_col="month_start",
        value_cols=["gnn", "rapid"],
        group_cols=["gauge_id", "comid"],
        scale=f"monthly_{args.monthly_aggregation}",
    )
    seasonal_metrics = _metric_rows(
        monthly[monthly["season_window"].isin(["Dec-May", "Jun-Sep"])],
        date_col="month_start",
        value_cols=["gnn", "rapid"],
        group_cols=["gauge_id", "comid", "season_window"],
        scale=f"seasonal_monthly_{args.monthly_aggregation}",
    )
    metrics = pd.concat([daily_metrics, monthly_metrics, seasonal_metrics], ignore_index=True)
    background_shape = _read_background_shape(args.background_shapefile)

    _write_tables(output_dir, aligned_daily, monthly, metrics, metadata, rapid_metadata, inferred_mapping)
    if inferred_mapping is not None:
        inferred_mapping.to_csv(output_dir / "rapid_inferred_comid_mapping.csv", index=False)
    _write_plots(
        output_dir,
        aligned_daily,
        monthly,
        metrics,
        metadata,
        args.dpi,
        args.monthly_aggregation,
        background_shape,
        args.map_dem,
        args.gauge_crs,
    )
    print(f"Wrote RAPID/GNN comparison analysis to {output_dir}")
    print(metrics[(metrics["scale"] == "daily")][["gauge_id", "comid", "model", "kge", "kgess", "nse", "rmse", "bias", "pbias"]].to_string(index=False))


if __name__ == "__main__":
    main()
