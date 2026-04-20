import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


TORCH_DTYPE_TO_NUMPY = {
    "torch.float16": np.float16,
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int16": np.int16,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
}


def load_basin_file(path: Path) -> List[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Basin list file not found: {path}")
    with path.open("r") as fp:
        basins = [line.strip() for line in fp if line.strip()]
    if not basins:
        raise ValueError(f"No basin IDs found in {path}")
    return basins


def get_available_ram_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)

    if hasattr(os, "sysconf"):
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)

    raise RuntimeError("Could not determine available system memory.")


def estimate_array_bytes(shape: Sequence[int], dtype: np.dtype) -> int:
    n_elem = 1
    for s in shape:
        n_elem *= int(s)
    return int(n_elem * np.dtype(dtype).itemsize)


def decide_io_mode(estimated_bytes: int, available_bytes: int, safety_factor: float = 0.65) -> str:
    usable = int(available_bytes * safety_factor)
    return "preload" if estimated_bytes <= usable else "lazy"


def infer_numpy_dtype(dtype_like) -> np.dtype:
    if isinstance(dtype_like, np.dtype):
        return dtype_like
    s = str(dtype_like)
    if s in TORCH_DTYPE_TO_NUMPY:
        return np.dtype(TORCH_DTYPE_TO_NUMPY[s])
    return np.dtype(s)


def _normalize_datetime_index(index_like) -> pd.DatetimeIndex:
    dt = pd.to_datetime(index_like)
    if isinstance(dt, pd.DatetimeIndex):
        return dt
    return pd.DatetimeIndex(dt)


def infer_time_coord(ds: xr.Dataset, preferred: Optional[str] = None) -> str:
    if preferred and preferred in ds.coords:
        return preferred
    candidates = [c for c in ds.coords if np.issubdtype(ds[c].dtype, np.datetime64)]
    if preferred and preferred in ds.dims:
        return preferred
    if candidates:
        return candidates[0]
    for candidate in ["time", "date", "datetime"]:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise RuntimeError("Could not infer time coordinate from dataset.")


def _discover_multi_file_paths(
    directory: Path,
    glob_pattern: str,
    *,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> List[Path]:
    root = Path(directory)
    direct_files = [path for path in root.glob(glob_pattern) if path.is_file()]

    nested_files: List[Path] = []
    try:
        subdirs = [path for path in root.iterdir() if path.is_dir()]
        has_subdirs = bool(subdirs)
    except FileNotFoundError:
        subdirs = []
        has_subdirs = False
    if has_subdirs:
        year_filtered_subdirs = subdirs
        if period_start is not None and period_end is not None:
            requested_years = set(range(int(period_start.year), int(period_end.year) + 1))
            parsed_year_subdirs = [path for path in subdirs if path.name.isdigit() and len(path.name) == 4]
            if parsed_year_subdirs:
                matching_year_subdirs = [
                    path for path in parsed_year_subdirs if int(path.name) in requested_years
                ]
                if matching_year_subdirs:
                    year_filtered_subdirs = matching_year_subdirs
                    logger.info(
                        "Restricting forcing file discovery to year folders %s for requested period %s -> %s",
                        sorted(path.name for path in matching_year_subdirs),
                        period_start,
                        period_end,
                    )
        nested_files = []
        for subdir in year_filtered_subdirs:
            nested_files.extend(path for path in subdir.rglob(glob_pattern) if path.is_file())

    files_by_path = {str(path): path for path in direct_files}
    for path in nested_files:
        files_by_path[str(path)] = path
    return [files_by_path[key] for key in sorted(files_by_path)]


def _infer_timestamp_from_forcing_path(path: Path) -> Optional[pd.Timestamp]:
    for token in re.findall(r"(?<!\d)(\d{8,14})(?!\d)", path.name):
        for fmt_length, fmt in (
            (14, "%Y%m%d%H%M%S"),
            (12, "%Y%m%d%H%M"),
            (10, "%Y%m%d%H"),
            (8, "%Y%m%d"),
        ):
            if len(token) != fmt_length:
                continue
            try:
                return pd.to_datetime(token, format=fmt)
            except Exception:
                continue
    return None


def _filter_forcing_files_by_period(
    files: Sequence[Path],
    *,
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> List[Path]:
    if period_start is None or period_end is None:
        return list(files)

    parsed: list[tuple[Path, pd.Timestamp]] = []
    unmatched: list[Path] = []
    for path in files:
        timestamp = _infer_timestamp_from_forcing_path(path)
        if timestamp is None:
            unmatched.append(path)
        else:
            parsed.append((path, timestamp))

    if not parsed:
        logger.info(
            "Could not infer timestamps from forcing filenames in %s files; loading all matched forcing files.",
            len(files),
        )
        return list(files)

    filtered = [path for path, timestamp in parsed if period_start <= timestamp <= period_end]
    if unmatched:
        logger.warning(
            "Could not infer timestamps for %s forcing files; keeping them to avoid accidental data loss.",
            len(unmatched),
        )
        filtered.extend(unmatched)

    filtered = sorted(filtered)
    logger.info(
        "Filtered forcing files by period %s -> %s: %s selected out of %s matched files.",
        period_start,
        period_end,
        len(filtered),
        len(files),
    )
    if not filtered:
        available_times = sorted(timestamp for _, timestamp in parsed)
        raise FileNotFoundError(
            "No forcing files fall within the requested period "
            f"{period_start} -> {period_end}. "
            f"Available forcing filename timestamps span "
            f"{available_times[0]} -> {available_times[-1]}."
        )
    return filtered


def _discover_yearly_zarr_stores(directory: Path) -> List[Path]:
    root = Path(directory)
    stores = [path for path in root.glob("*.zarr") if path.is_dir()]
    stores_by_path = {str(path): path for path in stores}
    return [stores_by_path[key] for key in sorted(stores_by_path)]


def _infer_year_from_zarr_store(path: Path) -> Optional[int]:
    match = re.fullmatch(r"(\d{4})\.zarr", path.name)
    if not match:
        return None
    return int(match.group(1))


def _select_yearly_zarr_stores_for_period(
    stores: Sequence[Path],
    *,
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> List[Path]:
    if period_start is None or period_end is None:
        return list(stores)

    requested_years = set(range(int(period_start.year), int(period_end.year) + 1))
    parsed: list[tuple[Path, int]] = []
    unmatched: list[Path] = []
    for path in stores:
        year = _infer_year_from_zarr_store(path)
        if year is None:
            unmatched.append(path)
        else:
            parsed.append((path, year))

    if not parsed:
        logger.info(
            "Could not infer yearly Zarr store names in %s stores; loading all matched stores.",
            len(stores),
        )
        return list(stores)

    selected = [path for path, year in parsed if year in requested_years]
    if unmatched:
        logger.warning(
            "Could not infer years for %s Zarr stores; keeping them to avoid accidental data loss.",
            len(unmatched),
        )
        selected.extend(unmatched)

    selected = sorted(selected)
    logger.info(
        "Filtered yearly Zarr stores by period %s -> %s: %s selected out of %s stores.",
        period_start,
        period_end,
        len(selected),
        len(stores),
    )
    if not selected:
        available_years = sorted(year for _, year in parsed)
        raise FileNotFoundError(
            "No yearly Zarr stores overlap the requested period "
            f"{period_start} -> {period_end}. "
            f"Available Zarr years span {available_years[0]} -> {available_years[-1]}."
        )
    return selected


def default_forcing_manifest_path(directory: Path) -> Path:
    return Path(directory) / ".routing_forcing_manifest.csv"


def _load_forcing_manifest_csv(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"path", "start_time", "end_time", "n_steps", "step_seconds"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Forcing manifest at {manifest_path} is missing columns: {sorted(missing)}")
    manifest["start_time"] = pd.to_datetime(manifest["start_time"])
    manifest["end_time"] = pd.to_datetime(manifest["end_time"])
    manifest["n_steps"] = manifest["n_steps"].astype(np.int64)
    manifest["step_seconds"] = pd.to_numeric(manifest["step_seconds"], errors="coerce")
    if "y_size" in manifest.columns:
        manifest["y_size"] = pd.to_numeric(manifest["y_size"], errors="coerce").astype("Int64")
    if "x_size" in manifest.columns:
        manifest["x_size"] = pd.to_numeric(manifest["x_size"], errors="coerce").astype("Int64")
    return manifest.sort_values("start_time").reset_index(drop=True)


def _cleanup_forcing_dataset(ds: xr.Dataset) -> xr.Dataset:
    drop_names: list[str] = []
    if "reference_time" in ds.variables:
        drop_names.append("reference_time")
    if "crs" in ds.data_vars:
        drop_names.append("crs")
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")
    return ds


def build_forcing_manifest(
    directory: Path,
    glob_pattern: str,
    *,
    manifest_path: Optional[Path] = None,
    open_kwargs: Optional[Dict] = None,
    preferred_time_dim: Optional[str] = None,
    show_progress: bool = False,
    progress_desc: str = "forcing manifest",
) -> pd.DataFrame:
    directory = Path(directory)
    open_kwargs = dict(open_kwargs or {})
    files = _discover_multi_file_paths(directory, glob_pattern)
    if not files:
        raise FileNotFoundError(f"No forcing files matched {glob_pattern} in {directory}")

    records: list[dict[str, object]] = []
    for file_path in tqdm(
        files,
        desc=progress_desc,
        total=len(files),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    ):
        ds = xr.open_dataset(file_path, **open_kwargs)
        try:
            ds = _cleanup_forcing_dataset(ds)
            time_dim = infer_time_coord(ds, preferred_time_dim)
            time_values = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].to_numpy()))
            if len(time_values) == 0:
                inferred_time = _infer_timestamp_from_forcing_path(file_path)
                if inferred_time is None:
                    raise RuntimeError(f"Could not infer forcing times for {file_path}")
                time_values = pd.DatetimeIndex([inferred_time])
            if len(time_values) <= 1:
                step_seconds = np.nan
            else:
                deltas = np.diff(time_values.view("i8")) / 1.0e9
                step_seconds = float(np.median(deltas))
            records.append(
                {
                    "path": str(file_path),
                    "start_time": time_values[0],
                    "end_time": time_values[-1],
                    "n_steps": int(len(time_values)),
                    "step_seconds": step_seconds,
                    "time_dim": str(time_dim),
                    "y_size": int(ds.sizes.get("y", ds.sizes.get("south_north", -1))),
                    "x_size": int(ds.sizes.get("x", ds.sizes.get("west_east", -1))),
                }
            )
        finally:
            ds.close()

    manifest = pd.DataFrame.from_records(records).sort_values("start_time").reset_index(drop=True)
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    return manifest


def load_or_build_forcing_manifest(
    directory: Path,
    glob_pattern: str,
    *,
    manifest_path: Optional[Path] = None,
    open_kwargs: Optional[Dict] = None,
    preferred_time_dim: Optional[str] = None,
    show_progress: bool = False,
    progress_desc: str = "forcing manifest",
    refresh: bool = False,
) -> pd.DataFrame:
    directory = Path(directory)
    manifest_path = Path(manifest_path) if manifest_path is not None else default_forcing_manifest_path(directory)
    if manifest_path.is_file() and not refresh:
        manifest = _load_forcing_manifest_csv(manifest_path)
        logger.info("Loaded forcing manifest from %s (%s files)", manifest_path, len(manifest))
        return manifest

    logger.info("Building forcing manifest at %s", manifest_path)
    return build_forcing_manifest(
        directory,
        glob_pattern,
        manifest_path=manifest_path,
        open_kwargs=open_kwargs,
        preferred_time_dim=preferred_time_dim,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def filter_forcing_manifest_by_period(
    manifest: pd.DataFrame,
    *,
    period_start: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if period_start is None or period_end is None:
        return manifest.copy().reset_index(drop=True)
    filtered = manifest.loc[
        (manifest["start_time"] <= period_end) & (manifest["end_time"] >= period_start)
    ].copy()
    if filtered.empty:
        raise FileNotFoundError(
            "No forcing files in the manifest overlap the requested period "
            f"{period_start} -> {period_end}. "
            f"Available forcing span is {manifest['start_time'].min()} -> {manifest['end_time'].max()}."
        )
    return filtered.sort_values("start_time").reset_index(drop=True)


def expand_forcing_manifest_time_index(manifest: pd.DataFrame) -> pd.DatetimeIndex:
    all_times: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        start_time = pd.Timestamp(row.start_time)
        n_steps = int(row.n_steps)
        if n_steps <= 0:
            continue
        if n_steps == 1:
            all_times.append(np.asarray([start_time.to_datetime64()]))
            continue
        step_seconds = float(row.step_seconds)
        if not np.isfinite(step_seconds) or step_seconds <= 0.0:
            total_seconds = (pd.Timestamp(row.end_time) - start_time).total_seconds()
            step_seconds = total_seconds / max(n_steps - 1, 1)
        freq = pd.to_timedelta(step_seconds, unit="s")
        all_times.append(pd.date_range(start_time, periods=n_steps, freq=freq).to_numpy())
    if not all_times:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate(all_times))


def select_forcing_manifest_rows_for_window(
    manifest: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    selected = manifest.loc[
        (manifest["start_time"] <= window_end) & (manifest["end_time"] >= window_start)
    ].copy()
    if selected.empty:
        raise FileNotFoundError(
            f"No forcing files overlap requested forcing window {window_start} -> {window_end}"
        )
    return selected.sort_values("start_time").reset_index(drop=True)


def open_forcing_dataset_from_files(
    files: Sequence[Path],
    *,
    variables: Sequence[str],
    open_kwargs: Optional[Dict] = None,
    preferred_time_dim: Optional[str] = None,
) -> xr.Dataset:
    open_kwargs = dict(open_kwargs or {})
    datasets = []
    concat_dim = None
    for file_path in files:
        ds = xr.open_dataset(file_path, **open_kwargs)
        ds = _cleanup_forcing_dataset(ds)
        if concat_dim is None:
            concat_dim = infer_time_coord(ds, preferred_time_dim)
        datasets.append(ds)

    ds_out = xr.concat(
        datasets,
        dim=concat_dim or "time",
        coords="minimal",
        compat="override",
        join="outer",
    )
    if concat_dim and concat_dim in ds_out.coords:
        ds_out = ds_out.sortby(concat_dim)

    missing = [v for v in variables if v not in ds_out.data_vars]
    if missing:
        raise KeyError(f"Variables not found in multi-file forcing dataset: {missing}")
    return ds_out[list(variables)]


def open_forcing_dataset_from_yearly_zarr(
    stores: Sequence[Path],
    *,
    variables: Sequence[str],
    open_kwargs: Optional[Dict] = None,
    preferred_time_dim: Optional[str] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> xr.Dataset:
    open_kwargs = dict(open_kwargs or {})
    datasets = []
    concat_dim = None
    for store_path in stores:
        ds = xr.open_zarr(store_path, **open_kwargs)
        ds = _cleanup_forcing_dataset(ds)
        if concat_dim is None:
            concat_dim = infer_time_coord(ds, preferred_time_dim)
        if period_start is not None and period_end is not None:
            ds = ds.sel({concat_dim: slice(period_start, period_end)})
        datasets.append(ds)

    if not datasets:
        raise FileNotFoundError("No yearly Zarr stores were selected for forcing input.")

    ds_out = xr.concat(
        datasets,
        dim=concat_dim or "time",
        coords="minimal",
        compat="override",
        join="outer",
    )
    if concat_dim and concat_dim in ds_out.coords:
        ds_out = ds_out.sortby(concat_dim)

    missing = [v for v in variables if v not in ds_out.data_vars]
    if missing:
        raise KeyError(f"Variables not found in yearly Zarr forcing dataset: {missing}")
    return ds_out[list(variables)]


def repeat_spinup_block(window: np.ndarray, spinup_length: int, base_period: int = 365) -> np.ndarray:
    if spinup_length <= 0:
        return np.empty((0, *window.shape[1:]), dtype=window.dtype)
    if window.shape[0] == 0:
        raise ValueError("Cannot create spinup from an empty window.")

    base_len = min(base_period, window.shape[0])
    base = window[:base_len]

    full_reps = spinup_length // base_len
    remainder = spinup_length % base_len

    parts = []
    if full_reps > 0:
        parts.extend([base] * full_reps)
    if remainder > 0:
        parts.append(base[:remainder])

    if not parts:
        return np.empty((0, *window.shape[1:]), dtype=window.dtype)

    return np.concatenate(parts, axis=0)


def load_csv_targets(
    file_path: Path,
    date_column: str,
    target_variables: Sequence[str],
    separator: str = ",",
    basin_id_column: Optional[str] = None,
    unit_conversion: str = "auto",
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Target file not found: {file_path}")

    df = pd.read_csv(file_path, skiprows=[1], sep=separator)

    def _resolve_date_column(frame: pd.DataFrame, preferred: str) -> tuple[str, pd.Series]:
        candidates: list[str] = []
        if preferred in frame.columns:
            candidates.append(preferred)

        lowered = {str(c).lower(): c for c in frame.columns}
        preferred_lower = str(preferred).lower()
        if preferred_lower in lowered and lowered[preferred_lower] not in candidates:
            candidates.append(lowered[preferred_lower])

        for candidate in ["datetime", "date", "dateTime", "Date"]:
            if candidate in frame.columns and candidate not in candidates:
                candidates.append(candidate)
            lowered_candidate = lowered.get(candidate.lower())
            if lowered_candidate is not None and lowered_candidate not in candidates:
                candidates.append(lowered_candidate)

        best_name = None
        best_values = None
        best_score = -1
        for candidate in candidates:
            parsed = pd.to_datetime(frame[candidate], errors="coerce")
            score = int(parsed.notna().sum())
            if score > best_score:
                best_name = candidate
                best_values = parsed
                best_score = score

        if best_name is None or best_values is None or best_score <= 0:
            raise KeyError(f"Could not resolve a valid date column from '{preferred}' in {file_path}")
        return str(best_name), best_values

    resolved_date_column, parsed_dates = _resolve_date_column(df, date_column)

    rename_map: Dict[str, str] = {}
    if "site_no" in df.columns:
        rename_map["site_no"] = "basin_id"

    discharge_source_column = None
    if "QQobs" not in df.columns:
        discharge_candidates = [
            column
            for column in df.columns
            if "00060" in str(column) and not str(column).lower().endswith("_cd")
        ]
        if discharge_candidates:
            discharge_source_column = str(discharge_candidates[0])
            rename_map[discharge_source_column] = "QQobs"

    if "Quality" not in df.columns:
        quality_candidates = [
            column
            for column in df.columns
            if str(column).lower() == "quality" or str(column).lower().endswith("_cd")
        ]
        if quality_candidates:
            rename_map[str(quality_candidates[0])] = "Quality"

    df = df.rename(columns=rename_map)
    df[resolved_date_column] = parsed_dates
    df = df.set_index(resolved_date_column).sort_index()
    missing_targets = [v for v in target_variables if v not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns in {file_path}: {missing_targets}")

    if basin_id_column and basin_id_column in df.columns:
        df = df.drop(columns=[basin_id_column])

    for target_name in target_variables:
        df[target_name] = pd.to_numeric(df[target_name], errors="coerce")

    conversion_key = str(unit_conversion or "auto").lower()
    should_convert_cfs = False
    if conversion_key in {"cfs_to_cms", "ft3s_to_m3s", "ft3/s_to_m3/s", "cubic_feet_to_cubic_meters"}:
        should_convert_cfs = True
    elif conversion_key == "auto":
        should_convert_cfs = discharge_source_column is not None and "00060" in discharge_source_column
    elif conversion_key in {"none", "identity"}:
        should_convert_cfs = False
    else:
        raise ValueError(
            f"Unsupported targets.unit_conversion '{unit_conversion}'. "
            "Use one of: auto, none, cfs_to_cms."
        )

    if should_convert_cfs:
        cfs_to_cms = 0.028316846592
        for target_name in target_variables:
            df[target_name] = df[target_name] * cfs_to_cms

    return df[list(target_variables)]


def save_scaler_yaml(scaler: Dict[str, Dict[str, float]], out_file: Path):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    with out_file.open("w") as fp:
        yaml.dump(scaler, fp)


def load_scaler_yaml(file_path: Path) -> Dict[str, Dict[str, float]]:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Scaler file not found: {file_path}")
    yaml = YAML(typ="safe")
    with file_path.open("r") as fp:
        data = yaml.load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scaler file must contain a mapping, got {type(data)!r}")
    return dict(data)


def open_dataset_from_mode(
    mode: str,
    variables: Sequence[str],
    directory: Optional[Path] = None,
    file_path: Optional[Path] = None,
    var_to_file_map: Optional[Dict[str, str]] = None,
    glob_pattern: str = "*.nc",
    open_kwargs: Optional[Dict] = None,
    dataset_kind: str = "forcing",
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> xr.Dataset:
    """Open a dataset according to a simple file-mode contract.

    Modes:
        - single_file: one NetCDF containing all variables.
        - multi_file: directory/glob of many NetCDFs.
        - per_variable: one file per variable.
        - multi_zarr_yearly: directory of yearly .zarr stores.
        - auto: infer from supplied arguments.

    Parameters
    ----------
    dataset_kind : str
        Controls how multi_file inputs are combined.
        - "forcing": concatenate files along time
        - "static": merge files by variable
    """
    open_kwargs = dict(open_kwargs or {})
    var_to_file_map = dict(var_to_file_map or {})

    if dataset_kind not in {"forcing", "static"}:
        raise ValueError(f"Unsupported dataset_kind: {dataset_kind}")

    def _open_single_dataset(path: Path) -> xr.Dataset:
        path = Path(path)
        mode_lower = str(mode).lower()
        is_zarr = mode_lower == "zarr" or path.suffix.lower() == ".zarr"
        if is_zarr:
            ds = xr.open_zarr(path, **open_kwargs)
        else:
            ds = xr.open_dataset(path, **open_kwargs)
        if dataset_kind == "forcing":
            ds = _cleanup_forcing_dataset(ds)
            if period_start is not None and period_end is not None:
                time_dim = infer_time_coord(ds, None)
                ds = ds.sel({time_dim: slice(period_start, period_end)})
        return ds

    if mode == "auto":
        if file_path:
            mode = "zarr" if Path(file_path).suffix.lower() == ".zarr" else "single_file"
        elif var_to_file_map:
            mode = "per_variable"
        else:
            mode = "multi_file"

    if mode in {"single_file", "zarr"}:
        if not file_path:
            raise ValueError(f"{mode} mode requires file_path")

        ds = _open_single_dataset(file_path)

        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Variables not found in {file_path}: {missing}")

        return ds[variables]

    if mode == "multi_file":
        if directory is None:
            raise ValueError("multi_file mode requires directory")

        files = _discover_multi_file_paths(
            Path(directory),
            glob_pattern,
            period_start=period_start,
            period_end=period_end,
        )
        if dataset_kind == "forcing":
            files = _filter_forcing_files_by_period(
                files,
                period_start=period_start,
                period_end=period_end,
            )
        if not files:
            raise FileNotFoundError(f"No files matched {glob_pattern} in {directory}")
        logger.info(
            "Resolved %s multi-file %s files from %s using pattern '%s'",
            len(files),
            dataset_kind,
            directory,
            glob_pattern,
        )

        # --------------------------------------------------
        # STATIC: many files = many variables on same grid
        # --------------------------------------------------
        if dataset_kind == "static":
            datasets = []
            for f in files:
                ds = xr.open_dataset(f, **open_kwargs)

                # remove stray scalar coords/vars that may break merge
                drop_names = []

                for c in ds.coords:
                    if c not in ("y", "x", "lat", "lon") and ds[c].ndim == 0:
                        drop_names.append(c)

                for v in ds.data_vars:
                    if v == "crs":
                        drop_names.append(v)

                if drop_names:
                    drop_names = sorted(set(drop_names))
                    ds = ds.drop_vars(drop_names, errors="ignore")

                datasets.append(ds)

            ds_out = xr.merge(datasets, compat="override", join="exact")

            missing = [v for v in variables if v not in ds_out.data_vars]
            if missing:
                raise KeyError(f"Variables not found in multi-file static dataset: {missing}")

            return ds_out[variables]

        # --------------------------------------------------
        # FORCING: many files = same vars across many timesteps
        # --------------------------------------------------
        if dataset_kind == "forcing":
            return open_forcing_dataset_from_files(
                files,
                variables=variables,
                open_kwargs=open_kwargs,
                preferred_time_dim="time",
            )

    if mode == "multi_zarr_yearly":
        if directory is None:
            raise ValueError("multi_zarr_yearly mode requires directory")
        if dataset_kind != "forcing":
            raise ValueError("multi_zarr_yearly mode is only supported for forcing datasets")

        stores = _discover_yearly_zarr_stores(Path(directory))
        if not stores:
            raise FileNotFoundError(f"No yearly Zarr stores matched '*.zarr' in {directory}")
        stores = _select_yearly_zarr_stores_for_period(
            stores,
            period_start=period_start,
            period_end=period_end,
        )
        logger.info(
            "Resolved %s yearly Zarr forcing stores from %s",
            len(stores),
            directory,
        )
        return open_forcing_dataset_from_yearly_zarr(
            stores,
            variables=variables,
            open_kwargs=open_kwargs,
            preferred_time_dim="time",
            period_start=period_start,
            period_end=period_end,
        )

    if mode == "per_variable":
        datasets = []
        for var in variables:
            if var not in var_to_file_map:
                raise KeyError(f"Variable '{var}' missing in var_to_file_map")

            ds_var = xr.open_dataset(var_to_file_map[var], **open_kwargs)

            # remove nuisance scalar coords/vars for static per-variable files
            drop_names = []

            for c in ds_var.coords:
                if c not in ("y", "x", "lat", "lon") and ds_var[c].ndim == 0:
                    drop_names.append(c)

            if "crs" in ds_var.data_vars:
                drop_names.append("crs")

            if drop_names:
                ds_var = ds_var.drop_vars(sorted(set(drop_names)), errors="ignore")

            if var not in ds_var.data_vars:
                alt_vars = list(ds_var.data_vars)
                if len(alt_vars) != 1:
                    raise KeyError(
                        f"File for variable '{var}' does not contain a matching variable name "
                        f"and has multiple candidates: {alt_vars}"
                    )
                ds_var = ds_var.rename({alt_vars[0]: var})

            datasets.append(ds_var[[var]])

        ds_out = xr.merge(datasets, compat="override", join="exact")
        return ds_out

    raise ValueError(f"Unsupported mode: {mode}")




# def open_dataset_from_mode(
#     mode: str,
#     variables: Sequence[str],
#     directory: Optional[Path] = None,
#     file_path: Optional[Path] = None,
#     var_to_file_map: Optional[Dict[str, str]] = None,
#     glob_pattern: str = "*.nc",
#     open_kwargs: Optional[Dict] = None,
# ) -> xr.Dataset:
#     """Open a dataset according to a simple file-mode contract.

#     Modes:
#         - single_file: one NetCDF containing all variables.
#         - multi_file: directory/glob of many NetCDFs to combine along coords.
#         - per_variable: one file per variable.
#         - auto: infer from supplied arguments.
#     """
#     open_kwargs = dict(open_kwargs or {})
#     var_to_file_map = dict(var_to_file_map or {})

#     if mode == "auto":
#         if file_path:
#             mode = "single_file"
#         elif var_to_file_map:
#             mode = "per_variable"
#         else:
#             mode = "multi_file"

#     if mode == "single_file":
#         if not file_path:
#             raise ValueError("single_file mode requires file_path")
#         ds = xr.open_dataset(file_path, **open_kwargs)
#         missing = [v for v in variables if v not in ds.data_vars]
#         if missing:
#             raise KeyError(f"Variables not found in {file_path}: {missing}")
#         return ds[variables]

#     if mode == "multi_file":
#         if directory is None:
#             raise ValueError("multi_file mode requires directory")
#         files = sorted(Path(directory).glob(glob_pattern))
#         if not files:
#             raise FileNotFoundError(f"No files matched {glob_pattern} in {directory}")
#         # ds = xr.open_mfdataset(files, combine="by_coords", **open_kwargs)
#         datasets = [xr.open_dataset(f, **open_kwargs) for f in files]
#         ds_static = xr.merge(datasets, compat="override", join="exact")
#         missing = [v for v in variables if v not in ds_static.data_vars]
#         if missing:
#             raise KeyError(f"Variables not found in multi-file dataset: {missing}")
#         return ds_static[variables]

#     if mode == "per_variable":
#         datasets = []
#         for var in variables:
#             if var not in var_to_file_map:
#                 raise KeyError(f"Variable '{var}' missing in var_to_file_map")
#             ds_var = xr.open_dataset(var_to_file_map[var], **open_kwargs)
#             if var not in ds_var.data_vars:
#                 alt_vars = list(ds_var.data_vars)
#                 if len(alt_vars) != 1:
#                     raise KeyError(
#                         f"File for variable '{var}' does not contain a matching variable name and has multiple candidates: {alt_vars}"
#                     )
#                 ds_var = ds_var.rename({alt_vars[0]: var})
#             datasets.append(ds_var[[var]])
#         return xr.merge(datasets)

#     raise ValueError(f"Unsupported mode: {mode}")
