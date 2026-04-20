import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lstm_gnn_routing.routing_models.graph_builder import build_routing_graph_payload, export_routing_graph_netcdf
from lstm_gnn_routing.routing_models.schema import normalize_routing_graph_payload
from lstm_gnn_routing.utils.config import RoutingConfig
from lstm_gnn_routing.utils.data import (
    decide_io_mode,
    estimate_array_bytes,
    expand_forcing_manifest_time_index,
    filter_forcing_manifest_by_period,
    get_available_ram_bytes,
    infer_numpy_dtype,
    infer_time_coord,
    load_basin_file,
    load_csv_targets,
    load_scaler_yaml,
    load_or_build_forcing_manifest,
    open_forcing_dataset_from_files,
    open_dataset_from_mode,
    repeat_spinup_block,
    save_scaler_yaml,
    select_forcing_manifest_rows_for_window,
)

logger = logging.getLogger(__name__)


lstm_gnn_routing_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _format_channel_value(value: Any) -> str:
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value)).replace(".", "p")
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value).replace(" ", "_").replace(".", "p")


def _build_channel_names(var_name: str, da: xr.DataArray, leading_dims: Sequence[str]) -> List[str]:
    if not leading_dims:
        return [var_name]

    coord_lists: List[List[str]] = []
    for dim in leading_dims:
        coord = da.coords.get(dim)
        if coord is None:
            coord_lists.append([f"{dim}{idx}" for idx in range(da.sizes[dim])])
            continue
        coord_lists.append([f"{dim}_{_format_channel_value(value)}" for value in coord.values])

    names = []
    for combination in itertools.product(*coord_lists):
        names.append(f"{var_name}__{'__'.join(combination)}")
    return names


def _expand_static_variable(
    var_name: str,
    da: xr.DataArray,
    *,
    dtype: np.dtype,
    y_dim: Optional[str] = None,
    x_dim: Optional[str] = None,
) -> tuple[np.ndarray, List[str], np.ndarray]:
    dims = list(da.dims)
    if len(dims) < 2:
        raise ValueError(f"Static variable '{var_name}' must have at least 2 dims, got {dims}")

    if y_dim is None or y_dim not in da.dims:
        y_dim = dims[-2]
    if x_dim is None or x_dim not in da.dims:
        x_dim = dims[-1]

    leading_dims = [dim for dim in da.dims if dim not in (y_dim, x_dim)]
    da = da.transpose(*leading_dims, y_dim, x_dim)
    arr = da.to_numpy().astype(dtype, copy=False)

    if not leading_dims:
        channel_array = arr[np.newaxis, ...]
        field_array = arr
    else:
        channel_array = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        field_array = np.moveaxis(channel_array, 0, -1)

    channel_names = _build_channel_names(var_name, da, leading_dims)
    return channel_array, channel_names, field_array


def _to_tensor_like(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        return tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
    if isinstance(value, dict):
        return {key: _to_tensor_like(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_tensor_like(item, dtype) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_time_feature_names(value: Any) -> list[str]:
    names = []
    for item in _as_list(value):
        text = str(item).strip().lower()
        if text:
            names.append(text)
    return names


def _load_graph_payload(file_path: Path, dtype: torch.dtype) -> Any:
    suffix = file_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(file_path, map_location="cpu")
    elif suffix in {".nc", ".nc4", ".netcdf"}:
        ds = xr.open_dataset(file_path)
        try:
            payload = {
                "edge_index": np.stack(
                    [
                        np.asarray(ds["edge_source"].values, dtype=np.int64),
                        np.asarray(ds["edge_target"].values, dtype=np.int64),
                    ],
                    axis=0,
                ),
                "flat_index": np.asarray(ds["flat_index"].values, dtype=np.int64),
                "node_y": np.asarray(ds["node_y"].values, dtype=np.int64),
                "node_x": np.asarray(ds["node_x"].values, dtype=np.int64),
            }
            if "edge_attr" in ds:
                payload["edge_attr"] = np.asarray(ds["edge_attr"].values, dtype=np.float32)
                if "edge_feature" in ds.coords:
                    payload["edge_feature_names"] = [str(value) for value in ds.coords["edge_feature"].values.tolist()]
            if "edge_weight" in ds:
                payload["edge_weight"] = np.asarray(ds["edge_weight"].values, dtype=np.float32)
            if "node_features" in ds:
                payload["node_features"] = np.asarray(ds["node_features"].values, dtype=np.float32)
            if "gauge_index" in ds:
                payload["gauge_index"] = np.asarray(ds["gauge_index"].values, dtype=np.int64)
            if "gauge_id" in ds:
                payload["gauge_ids"] = [str(value) for value in ds["gauge_id"].values.tolist()]
            if "runoff_target_index" in ds:
                payload["runoff_target_index"] = np.asarray(ds["runoff_target_index"].values, dtype=np.int64)
            if "runoff_source_index" in ds:
                payload["runoff_source_index"] = np.asarray(ds["runoff_source_index"].values, dtype=np.int64)
            if "runoff_source_flat_index" in ds:
                payload["runoff_source_flat_index"] = np.asarray(ds["runoff_source_flat_index"].values, dtype=np.int64)
            if "runoff_source_weight" in ds:
                payload["runoff_source_weight"] = np.asarray(ds["runoff_source_weight"].values, dtype=np.float32)
            if "runoff_source_features" in ds:
                payload["runoff_source_features"] = np.asarray(ds["runoff_source_features"].values, dtype=np.float32)
                if "runoff_source_feature" in ds.coords:
                    payload["runoff_source_feature_names"] = [
                        str(value) for value in ds.coords["runoff_source_feature"].values.tolist()
                    ]

            metadata: dict[str, Any] = {}
            for key, value in ds.attrs.items():
                if isinstance(value, str):
                    try:
                        metadata[key] = json.loads(value)
                        continue
                    except Exception:
                        pass
                metadata[key] = value
            if metadata:
                payload["metadata"] = metadata
        finally:
            ds.close()
    elif suffix == ".npz":
        with np.load(file_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
    elif suffix == ".npy":
        payload = np.load(file_path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            payload = payload.item()
    elif suffix in {".pkl", ".pickle"}:
        with file_path.open("rb") as fp:
            payload = pickle.load(fp)
    elif suffix == ".json":
        with file_path.open("r") as fp:
            payload = json.load(fp)
    else:
        raise ValueError(f"Unsupported routing graph file type: {file_path}")
    return _to_tensor_like(payload, dtype)


class RoutingDataset(Dataset):
    """Domain dataset for the standalone LSTM runoff + GNN routing workflow."""

    OPTIONAL_DYNAMIC_GROUPS = (
        ("x_forcing_ml", "ml", "forcing"),
        ("x_evolving_ml", "ml", "evolving"),
        ("x_routing_dynamic", "routing", "dynamic"),
    )

    OPTIONAL_STATIC_GROUPS = (
        ("x_static_ml", "ml", "static"),
        ("x_routing_static", "routing", "static"),
    )

    def __init__(self, config: RoutingConfig, period: str, scaler: Optional[Dict] = None):
        super().__init__()
        if period not in {"train", "validation", "test"}:
            raise ValueError("period must be one of: train, validation, test")

        self.config = config
        self.period = period
        self.scaler = dict(scaler or {})
        self._scaler_path = self._resolve_scaler_path()
        self._scaler_loaded_from_file = False
        if not self.scaler and self._scaler_path is not None and self._scaler_path.is_file():
            self.scaler = load_scaler_yaml(self._scaler_path)
            self._scaler_loaded_from_file = True
            logger.info("Loaded scaler file for period='%s': %s", self.period, self._scaler_path)
        self.dtype_np = infer_numpy_dtype(str(config.dtype).replace("torch.", ""))
        self.dtype_torch = config.dtype
        loading_cfg = config.section("loading")
        self.show_progress = bool(loading_cfg.get("show_progress", True))
        self.progress_leave = bool(loading_cfg.get("leave_progress", False))

        self.forcing_names = list(config.forcing_variables)
        self.static_names = list(config.static_variables)
        self.target_names = list(config.target_variables)

        self.sequence_length = int(config.sequence_length)
        self.spinup_length = int(config.spinup_length)
        self.stride = int(config.stride)
        self.window_sequence_years = config.window_sequence_years
        self.window_sequence_days = config.window_sequence_days
        self.window_stride_years = config.window_stride_years
        self.window_stride_days = config.window_stride_days
        self.window_spinup_years = config.window_spinup_years
        self.window_eval_periods = bool(config.windowing.get("apply_to_validation_test", False))
        self.runoff_model_type = str(config.runoff_model.get("type", "lstm")).lower()
        self.uses_physical_runoff = False
        routing_model_cfg = config.section("routing_model")
        self.routing_lag_context_days = int(routing_model_cfg.get("routing_lag_context_days", 0) or 0)
        runoff_model_cfg = config.section("runoff_model")
        self.runoff_warmup_days = int(
            runoff_model_cfg.get("warmup_days", runoff_model_cfg.get("lstm_warmup_days", 0)) or 0
        )
        self.prediction_context_days = max(self.routing_lag_context_days, self.runoff_warmup_days)

        self.basin_ids = self._load_basin_ids(period)
        self.period_start, self.period_end = self._get_period_dates(period)

        self.forcing_io_mode: Optional[str] = None
        self.forcing_ds: Optional[xr.Dataset] = None
        self.forcing_np: Optional[np.ndarray] = None
        self.forcing_manifest: Optional[pd.DataFrame] = None
        self.forcing_windowed_reads = bool(
            self.config.forcing_windowed_reads and self.config.forcing_file_mode == "multi_file"
        )
        self.static_np: Optional[np.ndarray] = None
        self.static_field_arrays: Dict[str, np.ndarray] = {}
        self.static_channel_map: Dict[str, List[str]] = {}
        self.targets_np: Optional[np.ndarray] = None
        self.target_scaler: Optional[Dict[str, Any]] = None
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.target_time_index: Optional[pd.DatetimeIndex] = None
        self.lookup: List[Dict[str, int]] = []
        self.x_info_global: Dict[str, Any] = {}

        self.optional_dynamic_groups: Dict[str, Dict[str, Any]] = {}
        self.optional_static_groups: Dict[str, Dict[str, Any]] = {}
        self.routing_graph: Any = None
        self.routing_compact_domain: bool = False
        self.routing_active_index: Optional[np.ndarray] = None
        self.routing_active_y_index: Optional[np.ndarray] = None
        self.routing_active_x_index: Optional[np.ndarray] = None

        self._load_all()

    def _progress(self, iterable, *, desc: str, total: Optional[int] = None):
        return tqdm(
            iterable,
            desc=f"{self.period}:{desc}",
            total=total,
            dynamic_ncols=True,
            leave=self.progress_leave,
            disable=not self.show_progress,
        )

    def _resolve_scaler_path(self) -> Optional[Path]:
        scaler_cfg = self.config.section("scaler")
        value = scaler_cfg.get("path", scaler_cfg.get("file_path"))
        if value in (None, ""):
            if self.config.normalize_data and self.config.save_scaler and self.config.run_dir:
                return Path(self.config.run_dir) / "routing_scaler.yml"
            return None
        return Path(value)

    def _should_save_scaler(self) -> bool:
        if self.period != "train":
            return False
        scaler_cfg = self.config.section("scaler")
        if bool(scaler_cfg.get("save", False)):
            return self._scaler_path is not None
        return bool(self.config.normalize_data and self.config.save_scaler and self._scaler_path is not None)

    def __len__(self):
        return len(self.lookup)

    def _resolve_spinup_spec(self, forcing_dates: pd.DatetimeIndex) -> tuple[int, int]:
        if self.window_spinup_years is None:
            return int(self.spinup_length), 365
        if forcing_dates.empty:
            return 0, 0

        if len(forcing_dates) > 1:
            step = forcing_dates[1] - forcing_dates[0]
        elif len(self.time_index) > 1:
            step = self.time_index[1] - self.time_index[0]
        else:
            step = pd.Timedelta(hours=1)

        start_time = pd.Timestamp(forcing_dates[0])
        end_exclusive = start_time + pd.DateOffset(years=int(self.window_spinup_years))
        step_ns = int(step.value)
        duration_ns = int((end_exclusive - start_time).value)
        spinup_length = max(int(round(duration_ns / step_ns)), 0) if step_ns > 0 else 0
        base_period = min(spinup_length, len(forcing_dates))
        return spinup_length, base_period

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.lookup[idx]

        f_start = item["forcing_start"]
        f_end = item["forcing_end"]
        t_start = item["target_start"]
        t_end = item["target_end"]
        runoff_context_start = self._context_start(f_start, self.prediction_context_days)
        routing_context_start = self._context_start(f_start, self.routing_lag_context_days)
        runoff_context_steps = self._context_steps(runoff_context_start, f_start)
        routing_context_steps = self._context_steps(routing_context_start, f_start)
        runoff_pre_routing_trim_steps = max(runoff_context_steps - routing_context_steps, 0)

        forcing_dates = self.time_index[f_start:f_end]
        if self.uses_physical_runoff:
            forcing_window = self._get_forcing_window(f_start, f_end)
        else:
            spatial_shape = (int(self.routing_active_index.size),) if self.routing_compact_domain and self.routing_active_index is not None else tuple(self.static_np.shape[-2:])
            forcing_window = np.empty((0, len(self.forcing_names), *spatial_shape), dtype=self.dtype_np)
        target_window = self.targets_np[t_start:t_end]
        target_mask = np.isfinite(target_window)

        spinup_length, spinup_base_period = self._resolve_spinup_spec(forcing_dates=pd.DatetimeIndex(forcing_dates))
        if not self.uses_physical_runoff:
            spinup_length = 0
            spinup_base_period = 0
        spinup_block = repeat_spinup_block(
            forcing_window,
            spinup_length,
            base_period=spinup_base_period,
        )

        if spinup_length > 0:
            forcing_total = np.concatenate([spinup_block, forcing_window], axis=0)
        else:
            forcing_total = forcing_window

        if self.routing_compact_domain:
            forcing_total = self._compact_routing_array(forcing_total)

        t_total = forcing_total.shape[0]
        t_forecast = forcing_window.shape[0]

        loss_mask = np.zeros((t_total,), dtype=np.bool_)
        loss_mask[-t_forecast:] = True
        spinup_mask = ~loss_mask

        target_dates = self.target_time_index[t_start:t_end]

        if spinup_length > 0:
            spin_dates = np.array([forcing_dates[0]] * spinup_length, dtype="datetime64[ns]")
            full_forcing_dates = np.concatenate([spin_dates, forcing_dates.to_numpy()])
        else:
            full_forcing_dates = forcing_dates.to_numpy()

        forcing_days = pd.DatetimeIndex(forcing_dates.normalize().unique())
        target_days = pd.DatetimeIndex(target_dates.normalize().unique())
        if not forcing_days.equals(target_days):
            raise RuntimeError(
                f"Forcing/target day mismatch in sample {idx}.\n"
                f"Forcing days: {len(forcing_days)}\n"
                f"Target days: {len(target_days)}"
            )

        x_info = {
            "forcing_names": list(self.forcing_names),
            "static_names": list(self.static_names),
            "target_names": list(self.target_names),
            "catchment_ids": list(self.basin_ids),
            "forcing_layout": "T,C,N" if self.routing_compact_domain else "T,C,Y,X",
            "static_layout": "C,N" if self.routing_compact_domain else "C,Y,X",
            "target_layout": "T,N,C",
            "target_normalized": self.target_scaler is not None,
            "target_transform": None if self.target_scaler is None else str(self.target_scaler["transform"]),
            "target_scaler_group": None if self.target_scaler is None else str(self.target_scaler["group"]),
            "time_index": full_forcing_dates,
            "forecast_time_index": forcing_dates.to_numpy(),
            "forcing_time_index": forcing_dates.to_numpy(),
            "target_time_index": target_dates.to_numpy(),
            "spinup_length": spinup_length,
            "spinup_years": self.window_spinup_years,
            "sequence_length": self.sequence_length,
            "window_sequence_years": self.window_sequence_years,
            "window_sequence_days": self.window_sequence_days,
            "window_stride_years": self.window_stride_years,
            "window_stride_days": self.window_stride_days,
            "forcing_sequence_length": len(forcing_dates),
            "target_sequence_length": len(target_dates),
            "runoff_warmup_days": self.runoff_warmup_days,
            "runoff_context_steps": runoff_context_steps,
            "routing_lag_context_days": self.routing_lag_context_days,
            "routing_context_steps": routing_context_steps,
            "runoff_pre_routing_trim_steps": runoff_pre_routing_trim_steps,
            "domain_shape": tuple(self.x_info_global.get("grid_shape", self.static_np.shape[-2:])),
            "routing_compact_domain": self.routing_compact_domain,
            "routing_active_flat_index": None if self.routing_active_index is None else self.routing_active_index.copy(),
            "routing_active_count": 0 if self.routing_active_index is None else int(self.routing_active_index.size),
            "forcing_io_mode": self.forcing_io_mode,
            **self.x_info_global,
        }

        sample: Dict[str, Any] = {
            "x_forcing_physical": torch.as_tensor(forcing_total, dtype=self.dtype_torch),
            "x_static_physical": torch.as_tensor(self.static_np, dtype=self.dtype_torch),
            "y": torch.as_tensor(target_window, dtype=self.dtype_torch),
            "target_mask": torch.as_tensor(target_mask, dtype=torch.bool),
            "loss_mask": torch.as_tensor(loss_mask),
            "spinup_mask": torch.as_tensor(spinup_mask),
            "prediction_context_steps": torch.as_tensor(routing_context_steps, dtype=torch.long),
            "routing_context_steps": torch.as_tensor(routing_context_steps, dtype=torch.long),
            "runoff_context_steps": torch.as_tensor(runoff_context_steps, dtype=torch.long),
            "runoff_pre_routing_trim_steps": torch.as_tensor(runoff_pre_routing_trim_steps, dtype=torch.long),
            "x_info": x_info,
        }

        for group_name, group in self.optional_dynamic_groups.items():
            dynamic_start = runoff_context_start if group_name == "x_forcing_ml" else f_start
            dynamic_window = self._get_dynamic_group_window(group, dynamic_start, f_end)
            if group.get("include_spinup", True):
                spinup_dynamic = repeat_spinup_block(
                    dynamic_window,
                    spinup_length,
                    base_period=spinup_base_period,
                )
                if spinup_length > 0:
                    dynamic_total = np.concatenate([spinup_dynamic, dynamic_window], axis=0)
                else:
                    dynamic_total = dynamic_window
            else:
                dynamic_total = dynamic_window
            sample[group_name] = torch.as_tensor(dynamic_total, dtype=self.dtype_torch)

        for group_name, group in self.optional_static_groups.items():
            sample[group_name] = torch.as_tensor(group["array"], dtype=self.dtype_torch)

        if self.routing_graph is not None:
            sample["routing_graph"] = self.routing_graph

        return sample

    def _load_basin_ids(self, period: str) -> List[str]:
        basin_file = {
            "train": self.config.train_basin_file,
            "validation": self.config.validation_basin_file,
            "test": self.config.test_basin_file,
        }[period]
        return load_basin_file(basin_file)

    def _get_period_dates(self, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        mapping = {
            "train": (self.config.train_start_date, self.config.train_end_date),
            "validation": (self.config.validation_start_date, self.config.validation_end_date),
            "test": (self.config.test_start_date, self.config.test_end_date),
        }
        return mapping[period]

    def _load_all(self):
        logger.info("Loading routing dataset for period='%s'", self.period)
        logger.info("Loading stage: static fields")
        self._load_static()
        logger.info("Loading stage: forcing data")
        self._open_or_preload_forcing()
        logger.info("Loading stage: streamflow targets")
        self._load_targets()
        logger.info("Loading stage: forcing/target time alignment")
        self._align_time_and_slice_period()
        logger.info("Loading stage: target scaling")
        self._normalize_targets_if_requested()
        logger.info("Loading stage: optional groups and routing graph")
        self._load_optional_groups()
        logger.info("Loading stage: sample lookup")
        self._build_lookup()

        logger.info(
            "Finished dataset period='%s' | forcing_steps=%s | target_steps=%s | samples=%s",
            self.period,
            len(self.time_index) if self.time_index is not None else 0,
            len(self.target_time_index) if self.target_time_index is not None else 0,
            len(self.lookup),
        )

        if self._should_save_scaler():
            if self._scaler_path is None:
                raise RuntimeError("Internal error: scaler save requested but no scaler path was resolved.")
            save_scaler_yaml(self.scaler, self._scaler_path)
            logger.info("Saved scaler file: %s", self._scaler_path)

    def _load_static(self):
        group = self._load_static_group_from_spec(
            self.config.static,
            group_name="x_static_physical",
            normalize=False,
            default_y_dim="y",
            default_x_dim="x",
        )
        self.static_np = group["array"]
        self.static_field_arrays = group["field_arrays"]
        self.static_channel_map = group["channel_map"]

        lat2d = group["lat2d"]
        lon2d = group["lon2d"]
        if lat2d is None or lon2d is None:
            y_size, x_size = self.static_np.shape[-2:]
            yy, xx = np.meshgrid(np.arange(y_size), np.arange(x_size), indexing="ij")
            lat2d = yy.astype(np.float32)
            lon2d = xx.astype(np.float32)
        self.x_info_global["static_dims"] = ("y", "x")
        self.x_info_global["grid_y"] = np.arange(lat2d.shape[0], dtype=np.int32)
        self.x_info_global["grid_x"] = np.arange(lon2d.shape[1], dtype=np.int32)
        self.x_info_global["lat2d"] = lat2d
        self.x_info_global["lon2d"] = lon2d
        self.x_info_global["proj_y2d"] = group.get("y2d")
        self.x_info_global["proj_x2d"] = group.get("x2d")
        self.x_info_global["grid_shape"] = lat2d.shape
        self.x_info_global["static_channel_names"] = list(group["names"])

    def _open_or_preload_forcing(self):
        if self.forcing_windowed_reads:
            self._prepare_windowed_forcing()
            return

        logger.info(
            "Opening forcing data from %s with mode='%s', pattern='%s', period=%s -> %s",
            self.config.forcing_dir or self.config.forcing_file_path,
            self.config.forcing_file_mode,
            self.config.forcing_glob,
            self.period_start,
            self.period_end,
        )
        self.forcing_ds = open_dataset_from_mode(
            mode=self.config.forcing_file_mode,
            variables=self.forcing_names,
            directory=self.config.forcing_dir,
            file_path=self.config.forcing_file_path,
            var_to_file_map=self.config.forcing_var_to_file_map,
            glob_pattern=self.config.forcing_glob,
            open_kwargs=self.config.forcing_open_kwargs,
            dataset_kind="forcing",
            period_start=self.period_start,
            period_end=self.period_end,
        )

        time_dim = infer_time_coord(self.forcing_ds, self.config.forcing_time_dim)
        y_dim = self.config.forcing_y_dim
        x_dim = self.config.forcing_x_dim
        if y_dim not in self.forcing_ds.dims or x_dim not in self.forcing_ds.dims:
            raise KeyError(
                f"Expected forcing spatial dims '{y_dim}', '{x_dim}' not found in dataset dims {self.forcing_ds.dims}"
            )

        n_time = int(self.forcing_ds.sizes[time_dim])
        n_y = int(self.forcing_ds.sizes[y_dim])
        n_x = int(self.forcing_ds.sizes[x_dim])
        n_vars = len(self.forcing_names)

        estimated = estimate_array_bytes((n_time, n_vars, n_y, n_x), self.dtype_np)
        available = get_available_ram_bytes()
        auto_choice = decide_io_mode(estimated, available, self.config.memory_safety_factor)

        self.forcing_io_mode = auto_choice if self.config.io_mode == "auto" else self.config.io_mode
        self.time_coord_name = time_dim
        self.forcing_dims = (time_dim, y_dim, x_dim)
        self.forcing_name_to_index = {name: idx for idx, name in enumerate(self.forcing_names)}

        if self.forcing_io_mode == "preload":
            arrays = []
            for var in self._progress(
                self.forcing_names,
                desc="forcing preload vars",
                total=len(self.forcing_names),
            ):
                da = self.forcing_ds[var].transpose(time_dim, y_dim, x_dim)
                arrays.append(da.to_numpy().astype(self.dtype_np, copy=False))
            self.forcing_np = np.stack(arrays, axis=1)
        else:
            self.forcing_np = None

        self.time_index = pd.DatetimeIndex(pd.to_datetime(self.forcing_ds[time_dim].to_numpy()))
        self.x_info_global["forcing_dims"] = (time_dim, y_dim, x_dim)
        logger.info(
            "Forcing ready | io_mode=%s | shape=[T=%s,C=%s,Y=%s,X=%s]",
            self.forcing_io_mode,
            n_time,
            n_vars,
            n_y,
            n_x,
        )

    def _prepare_windowed_forcing(self):
        logger.info(
            "Preparing windowed forcing reads from %s with manifest=%s, pattern='%s', period=%s -> %s",
            self.config.forcing_dir,
            self.config.forcing_manifest_path,
            self.config.forcing_glob,
            self.period_start,
            self.period_end,
        )
        manifest = load_or_build_forcing_manifest(
            self.config.forcing_dir,
            self.config.forcing_glob,
            manifest_path=self.config.forcing_manifest_path,
            open_kwargs=self.config.forcing_open_kwargs,
            preferred_time_dim=self.config.forcing_time_dim,
            show_progress=self.show_progress,
            progress_desc=f"{self.period}:forcing manifest",
        )
        manifest = filter_forcing_manifest_by_period(
            manifest,
            period_start=self.period_start,
            period_end=self.period_end,
        )
        if manifest.empty:
            raise RuntimeError(
                f"No forcing files overlap requested period {self.period_start} -> {self.period_end}"
            )

        self.forcing_manifest = manifest.reset_index(drop=True)
        self.forcing_ds = None
        self.forcing_np = None
        self.forcing_io_mode = "windowed"
        self.time_coord_name = self.config.forcing_time_dim
        self.forcing_dims = (self.config.forcing_time_dim, self.config.forcing_y_dim, self.config.forcing_x_dim)
        self.forcing_name_to_index = {name: idx for idx, name in enumerate(self.forcing_names)}
        self.time_index = expand_forcing_manifest_time_index(self.forcing_manifest)

        sample_file = Path(str(self.forcing_manifest.iloc[0]["path"]))
        sample_ds = xr.open_dataset(sample_file, **self.config.forcing_open_kwargs)
        try:
            time_dim = infer_time_coord(sample_ds, self.config.forcing_time_dim)
            y_dim = self.config.forcing_y_dim
            x_dim = self.config.forcing_x_dim
            if y_dim not in sample_ds.dims or x_dim not in sample_ds.dims:
                raise KeyError(
                    f"Expected forcing spatial dims '{y_dim}', '{x_dim}' not found in sample forcing file dims {sample_ds.dims}"
                )
            self.forcing_dims = (time_dim, y_dim, x_dim)
            for name in self.forcing_names:
                if name not in sample_ds.data_vars:
                    raise KeyError(f"Variable '{name}' not found in windowed forcing sample file {sample_file}")
            n_y = int(sample_ds.sizes[y_dim])
            n_x = int(sample_ds.sizes[x_dim])
        finally:
            sample_ds.close()

        logger.info(
            "Forcing ready | io_mode=%s | files=%s | shape=[T=%s,C=%s,Y=%s,X=%s]",
            self.forcing_io_mode,
            len(self.forcing_manifest),
            len(self.time_index),
            len(self.forcing_names),
            n_y,
            n_x,
        )

    def _load_targets(self):
        logger.info(
            "Loading target CSVs for %s gauges from %s",
            len(self.basin_ids),
            self.config.target_dir,
        )
        target_frames = []
        base_dir = self.config.target_dir
        for basin_id in self._progress(
            self.basin_ids,
            desc="target basins",
            total=len(self.basin_ids),
        ):
            file_path = Path(base_dir) / self.config.target_file_pattern.format(basin_id=basin_id)
            df = load_csv_targets(
                file_path=file_path,
                date_column=self.config.target_date_column,
                target_variables=self.target_names,
                separator=self.config.target_separator,
                basin_id_column=self.config.target_basin_id_column,
                unit_conversion=self.config.target_unit_conversion,
            )
            target_frames.append(df)

        if not target_frames:
            raise RuntimeError("No target files were loaded.")

        combined_index = target_frames[0].index
        for df in target_frames[1:]:
            combined_index = combined_index.union(df.index)
        if combined_index.empty:
            raise RuntimeError("Target files do not contain any timestamps.")
        combined_index = combined_index.sort_values()

        arrays = []
        for df in self._progress(
            target_frames,
            desc="target arrays",
            total=len(target_frames),
        ):
            arrays.append(df.reindex(combined_index)[self.target_names].to_numpy().astype(self.dtype_np, copy=False))

        self.target_time_index = pd.DatetimeIndex(combined_index)
        self.targets_np = np.stack(arrays, axis=1)
        logger.info(
            "Targets ready | shape=[T=%s,N=%s,C=%s]",
            self.targets_np.shape[0],
            self.targets_np.shape[1],
            self.targets_np.shape[2],
        )

    def _align_time_and_slice_period(self):
        forcing_in_period = self.time_index[
            (self.time_index >= self.period_start) & (self.time_index <= self.period_end)
        ]
        target_in_period = self.target_time_index[
            (self.target_time_index >= self.period_start) & (self.target_time_index <= self.period_end)
        ]

        if len(forcing_in_period) == 0:
            raise RuntimeError(
                f"No forcing time steps found for period {self.period}: "
                f"{self.period_start} to {self.period_end}"
            )
        if len(target_in_period) == 0:
            raise RuntimeError(
                f"No target time steps found for period {self.period}: "
                f"{self.period_start} to {self.period_end}"
            )

        forcing_days = pd.DatetimeIndex(forcing_in_period.normalize().unique())
        target_days = pd.DatetimeIndex(target_in_period.normalize().unique())
        common_days = forcing_days.intersection(target_days)

        if common_days.empty:
            raise RuntimeError(
                f"No overlapping forcing/target days found for period {self.period}: "
                f"{self.period_start} to {self.period_end}"
            )

        forcing_mask = (
            (self.time_index >= self.period_start)
            & (self.time_index <= self.period_end)
            & (self.time_index.normalize().isin(common_days))
        )
        target_mask = (
            (self.target_time_index >= self.period_start)
            & (self.target_time_index <= self.period_end)
            & (self.target_time_index.normalize().isin(common_days))
        )

        if self.forcing_manifest is not None:
            self.time_index = pd.DatetimeIndex(self.time_index[forcing_mask])
        elif self.forcing_np is not None:
            self.forcing_np = self.forcing_np[forcing_mask]
        else:
            self.forcing_ds = self.forcing_ds.isel({self.time_coord_name: forcing_mask})

        self.targets_np = self.targets_np[target_mask]
        if self.forcing_manifest is None:
            self.time_index = pd.DatetimeIndex(self.time_index[forcing_mask])
        self.target_time_index = pd.DatetimeIndex(self.target_time_index[target_mask])

        forcing_days_final = pd.DatetimeIndex(self.time_index.normalize().unique())
        target_days_final = pd.DatetimeIndex(self.target_time_index.normalize().unique())
        if not forcing_days_final.equals(target_days_final):
            raise RuntimeError(
                "Forcing and target days diverged after alignment.\n"
                f"Forcing days: {forcing_days_final[:5]} ... total={len(forcing_days_final)}\n"
                f"Target days: {target_days_final[:5]} ... total={len(target_days_final)}"
            )

    def _target_scaler_group(self) -> str:
        targets_cfg = self.config.section("targets")
        return str(targets_cfg.get("scaler_group", targets_cfg.get("scaler_name", "streamflow")))

    def _target_transform_name(self) -> str:
        targets_cfg = self.config.section("targets")
        return str(targets_cfg.get("transform", "identity")).lower()

    def _target_normalization_enabled(self) -> bool:
        return bool(self.config.section("targets").get("normalize", False))

    def _transform_targets_array(self, array: np.ndarray, transform: str) -> np.ndarray:
        transform = str(transform or "identity").lower()
        if transform in {"", "none", "identity"}:
            return array.astype(self.dtype_np, copy=True)
        if transform == "log1p":
            return np.log1p(np.clip(array.astype(self.dtype_np, copy=True), a_min=0.0, a_max=None))
        raise ValueError(f"Unsupported targets.transform '{transform}'. Use one of: identity, log1p.")

    def _inverse_transform_targets_array(self, array: np.ndarray, transform: str) -> np.ndarray:
        transform = str(transform or "identity").lower()
        if transform in {"", "none", "identity"}:
            return array
        if transform == "log1p":
            return np.expm1(array)
        raise ValueError(f"Unsupported targets.transform '{transform}'. Use one of: identity, log1p.")

    def _target_stats_available(self, group_name: str) -> bool:
        stats = (self.scaler.get("target_stats") or {}).get(group_name)
        if not stats:
            return False
        means = stats.get("means") or {}
        stds = stats.get("stds") or {}
        for basin_id in self.basin_ids:
            if str(basin_id) not in means or str(basin_id) not in stds:
                return False
            for target_name in self.target_names:
                if target_name not in (means[str(basin_id)] or {}) or target_name not in (stds[str(basin_id)] or {}):
                    return False
        return True

    def _load_target_scaler(self, group_name: str) -> tuple[np.ndarray, np.ndarray, str]:
        stats = (self.scaler.get("target_stats") or {}).get(group_name)
        if not stats:
            raise ValueError(
                f"Missing target scaler stats for '{group_name}'. "
                "Compute/load the training scaler before validation/test."
            )
        transform = str(stats.get("transform", self._target_transform_name())).lower()
        means = np.zeros((len(self.basin_ids), len(self.target_names)), dtype=self.dtype_np)
        stds = np.ones_like(means)
        means_map = stats.get("means") or {}
        stds_map = stats.get("stds") or {}
        for basin_idx, basin_id in enumerate(self.basin_ids):
            basin_key = str(basin_id)
            if basin_key not in means_map or basin_key not in stds_map:
                raise ValueError(f"Target scaler '{group_name}' is missing gauge '{basin_key}'")
            for target_idx, target_name in enumerate(self.target_names):
                try:
                    means[basin_idx, target_idx] = float(means_map[basin_key][target_name])
                    stds[basin_idx, target_idx] = max(float(stds_map[basin_key][target_name]), 1.0e-6)
                except KeyError as exc:
                    raise ValueError(
                        f"Target scaler '{group_name}' is missing variable '{target_name}' for gauge '{basin_key}'"
                    ) from exc
        return means, stds, transform

    def _compute_target_scaler(self, group_name: str, transform: str) -> tuple[np.ndarray, np.ndarray]:
        if self.targets_np is None:
            raise RuntimeError("Targets must be loaded before computing target scaler.")
        transformed = self._transform_targets_array(self.targets_np, transform)
        means = np.nanmean(transformed, axis=0).astype(self.dtype_np, copy=False)
        stds = np.nanstd(transformed, axis=0).astype(self.dtype_np, copy=False)
        means = np.where(np.isfinite(means), means, 0.0).astype(self.dtype_np, copy=False)
        stds = np.where(np.isfinite(stds) & (stds > 1.0e-6), stds, 1.0).astype(self.dtype_np, copy=False)

        self.scaler.setdefault("target_stats", {})[group_name] = {
            "transform": transform,
            "basin_ids": [str(value) for value in self.basin_ids],
            "variables": list(self.target_names),
            "means": {
                str(basin_id): {
                    target_name: float(means[basin_idx, target_idx])
                    for target_idx, target_name in enumerate(self.target_names)
                }
                for basin_idx, basin_id in enumerate(self.basin_ids)
            },
            "stds": {
                str(basin_id): {
                    target_name: float(stds[basin_idx, target_idx])
                    for target_idx, target_name in enumerate(self.target_names)
                }
                for basin_idx, basin_id in enumerate(self.basin_ids)
            },
        }
        return means, stds

    def _normalize_targets_if_requested(self) -> None:
        if not self._target_normalization_enabled():
            self.target_scaler = None
            return
        if self.targets_np is None:
            raise RuntimeError("Targets must be loaded before normalization.")

        group_name = self._target_scaler_group()
        requested_transform = self._target_transform_name()
        if self._target_stats_available(group_name):
            means, stds, transform = self._load_target_scaler(group_name)
        else:
            if self.period != "train":
                raise ValueError(
                    f"targets.normalize=true requires training target stats for '{group_name}' "
                    f"before loading period='{self.period}'."
                )
            transform = requested_transform
            means, stds = self._compute_target_scaler(group_name, transform)

        transformed = self._transform_targets_array(self.targets_np, transform)
        self.targets_np = ((transformed - means[None, :, :]) / stds[None, :, :]).astype(self.dtype_np, copy=False)
        self.target_scaler = {
            "group": group_name,
            "transform": transform,
            "means": means,
            "stds": stds,
        }
        logger.info(
            "Targets normalized | group=%s | transform=%s | gauges=%s | variables=%s",
            group_name,
            transform,
            len(self.basin_ids),
            ",".join(self.target_names),
        )

    def inverse_transform_targets_array(self, array: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return array
        values = np.asarray(array, dtype=self.dtype_np)
        means = np.asarray(self.target_scaler["means"], dtype=self.dtype_np)
        stds = np.asarray(self.target_scaler["stds"], dtype=self.dtype_np)

        if values.ndim >= 1 and means.shape[1] == 1 and values.shape[-1] == means.shape[0]:
            mean_values = means[:, 0]
            std_values = stds[:, 0]
        elif values.ndim >= 2 and tuple(values.shape[-2:]) == tuple(means.shape):
            mean_values = means
            std_values = stds
        else:
            raise ValueError(
                "Cannot inverse-transform targets with shape "
                f"{tuple(values.shape)} using scaler shape {tuple(means.shape)}"
            )

        while mean_values.ndim < values.ndim:
            mean_values = np.expand_dims(mean_values, axis=0)
            std_values = np.expand_dims(std_values, axis=0)

        physical = values * std_values + mean_values
        physical = self._inverse_transform_targets_array(physical, str(self.target_scaler["transform"]))
        return physical.astype(self.dtype_np, copy=False)

    def _time_feature_values(self, time_index: pd.DatetimeIndex, features: Sequence[str]) -> np.ndarray:
        if not features:
            return np.empty((len(time_index), 0), dtype=self.dtype_np)
        time_index = pd.DatetimeIndex(time_index)
        arrays = []
        for feature in features:
            name = str(feature).lower()
            if name in {"doy_sin", "dayofyear_sin"}:
                values = np.sin(2.0 * np.pi * (time_index.dayofyear.to_numpy(dtype=np.float64) - 1.0) / 366.0)
            elif name in {"doy_cos", "dayofyear_cos"}:
                values = np.cos(2.0 * np.pi * (time_index.dayofyear.to_numpy(dtype=np.float64) - 1.0) / 366.0)
            elif name == "month_sin":
                values = np.sin(2.0 * np.pi * (time_index.month.to_numpy(dtype=np.float64) - 1.0) / 12.0)
            elif name == "month_cos":
                values = np.cos(2.0 * np.pi * (time_index.month.to_numpy(dtype=np.float64) - 1.0) / 12.0)
            else:
                raise ValueError(
                    f"Unsupported time feature '{feature}'. "
                    "Use one of: doy_sin, doy_cos, month_sin, month_cos."
                )
            arrays.append(values.astype(self.dtype_np, copy=False))
        return np.stack(arrays, axis=1).astype(self.dtype_np, copy=False)

    def _append_time_features(
        self,
        array: np.ndarray,
        time_index: pd.DatetimeIndex,
        features: Sequence[str],
    ) -> np.ndarray:
        if not features:
            return array
        values = self._time_feature_values(time_index, features)
        if values.shape[0] != array.shape[0]:
            raise ValueError(
                f"Time feature length {values.shape[0]} does not match dynamic array length {array.shape[0]}"
            )
        spatial_shape = tuple(int(v) for v in array.shape[2:])
        broadcast_shape = (values.shape[0], values.shape[1]) + (1,) * len(spatial_shape)
        tile_shape = (1, 1) + spatial_shape
        feature_array = np.tile(values.reshape(broadcast_shape), tile_shape).astype(self.dtype_np, copy=False)
        return np.concatenate([array, feature_array], axis=1)

    def _load_optional_groups(self):
        for group_name, section_name, subsection_name in self._progress(
            self.OPTIONAL_DYNAMIC_GROUPS,
            desc="optional dynamic groups",
            total=len(self.OPTIONAL_DYNAMIC_GROUPS),
        ):
            spec = self._get_nested_optional_spec(section_name, subsection_name)
            if not spec or not spec.get("variables"):
                continue
            normalize = bool(spec.get("normalize", self.config.normalize_data if section_name == "ml" else False))
            self.optional_dynamic_groups[group_name] = self._load_dynamic_group_from_spec(
                spec,
                group_name=group_name,
                normalize=normalize,
            )
            self.x_info_global[f"{group_name}_names"] = list(self.optional_dynamic_groups[group_name]["names"])
            self.x_info_global[f"{group_name}_frequency"] = self.optional_dynamic_groups[group_name].get("frequency", "native")

        for group_name, section_name, subsection_name in self._progress(
            self.OPTIONAL_STATIC_GROUPS,
            desc="optional static groups",
            total=len(self.OPTIONAL_STATIC_GROUPS),
        ):
            spec = self._get_nested_optional_spec(section_name, subsection_name)
            if not spec:
                continue
            normalize = bool(spec.get("normalize", self.config.normalize_data if section_name == "ml" else False))
            if not spec.get("variables"):
                continue
            self.optional_static_groups[group_name] = self._load_static_group_from_spec(
                spec,
                group_name=group_name,
                normalize=normalize,
            )
            self.x_info_global[f"{group_name}_names"] = list(self.optional_static_groups[group_name]["names"])
            if group_name == "x_routing_static":
                self.x_info_global["routing_proj_y2d"] = self.optional_static_groups[group_name].get("y2d")
                self.x_info_global["routing_proj_x2d"] = self.optional_static_groups[group_name].get("x2d")

        routing_cfg = self.config.routing
        graph_spec = routing_cfg.get("graph", {}) if isinstance(routing_cfg, dict) else {}
        if graph_spec:
            self._load_routing_graph(graph_spec)

    def _load_routing_graph(self, graph_spec: Mapping[str, Any]):
        graph_options = dict(graph_spec.get("options", {}) or {})
        edge_feature_spec = dict(graph_spec.get("edge_features", {}) or {})
        graph_mode = str(
            graph_spec.get(
                "mode",
                "load" if graph_spec.get("file_path") else "build" if graph_spec.get("builder") else "load",
            )
        ).lower()

        if graph_mode not in {"load", "build"}:
            raise ValueError("routing.graph.mode must be either 'load' or 'build'")

        raw_graph: Any
        if graph_mode == "load":
            file_path = graph_spec.get("file_path", graph_options.get("file_path"))
            if not file_path:
                raise ValueError("routing.graph.mode='load' requires routing.graph.file_path")
            logger.info("Loading routing graph from %s", file_path)
            raw_graph = _load_graph_payload(Path(file_path), self.dtype_torch)
        else:
            cache_path = graph_spec.get("cache_path", graph_options.get("cache_path"))
            artifact_netcdf_path = graph_spec.get("artifact_netcdf_path", graph_options.get("artifact_netcdf_path"))
            overwrite_cache = bool(graph_spec.get("overwrite_cache", graph_options.get("overwrite_cache", False)))
            requested_edge_features = list(edge_feature_spec.get("derived", []) or [])
            requested_weight_feature = edge_feature_spec.get("weight_feature")
            requested_weight_normalization = edge_feature_spec.get("weight_normalization")
            use_cached_graph = False
            if cache_path and Path(cache_path).is_file() and not overwrite_cache:
                logger.info("Loading cached built routing graph from %s", cache_path)
                cached_graph = _load_graph_payload(Path(cache_path), self.dtype_torch)
                cached_metadata = dict(cached_graph.get("metadata", {}) or {}) if isinstance(cached_graph, Mapping) else {}
                cached_edge_features = list(cached_graph.get("edge_feature_names", []) or cached_metadata.get("derived_edge_features", []) or [])
                cached_weight_feature = cached_metadata.get("edge_weight_feature")
                cached_weight_normalization = cached_metadata.get("edge_weight_normalization")
                cached_builder = cached_metadata.get("builder")
                requested_builder = str(graph_spec.get("builder", "grid_4_neighbor")).lower()
                use_cached_graph = (
                    cached_builder == requested_builder
                    and cached_edge_features == requested_edge_features
                    and cached_weight_feature == requested_weight_feature
                    and cached_weight_normalization == requested_weight_normalization
                )
                if use_cached_graph:
                    raw_graph = cached_graph
                else:
                    logger.info(
                        "Cached routing graph is stale for requested edge features; rebuilding graph "
                        "(cached features=%s, requested=%s, cached weight=%s, requested weight=%s)",
                        cached_edge_features,
                        requested_edge_features,
                        cached_weight_feature,
                        requested_weight_feature,
                    )
            if not use_cached_graph:
                mask_spec = graph_spec.get("mask")
                elevation_spec = graph_spec.get("elevation")
                flow_spec = graph_spec.get("flow_direction")
                flowline_spec = graph_spec.get("flowlines", graph_spec.get("flowline"))
                logger.info(
                    "Building routing graph with builder '%s'%s",
                    graph_spec.get("builder", "grid_4_neighbor"),
                    f" and caching to {cache_path}" if cache_path else "",
                )
                raw_graph = build_routing_graph_payload(
                    builder=str(graph_spec.get("builder", "grid_4_neighbor")),
                    grid_shape=self.x_info_global.get("grid_shape"),
                    mask_array=self._resolve_routing_array_spec(mask_spec, required=False),
                    mask_spec=mask_spec if isinstance(mask_spec, Mapping) else None,
                    elevation_array=self._resolve_routing_array_spec(elevation_spec, required=False),
                    flow_direction=self._resolve_routing_array_spec(flow_spec, required=False),
                    flow_direction_encoding=str(
                        graph_spec.get("flow_direction_encoding", graph_options.get("flow_direction_encoding", "arcgis"))
                    ),
                    flowlines=flowline_spec if isinstance(flowline_spec, Mapping) else None,
                    node_feature_array=self._resolve_routing_node_feature_spec(
                        graph_spec.get("node_features"),
                        required=False,
                    ),
                    derived_edge_features=tuple(requested_edge_features),
                    edge_weight_feature=requested_weight_feature,
                    edge_weight_normalization=requested_weight_normalization,
                    gauges=graph_spec.get("gauges"),
                    lat2d=self.x_info_global.get("lat2d"),
                    lon2d=self.x_info_global.get("lon2d"),
                    y2d=self.x_info_global.get("routing_proj_y2d", self.x_info_global.get("proj_y2d")),
                    x2d=self.x_info_global.get("routing_proj_x2d", self.x_info_global.get("proj_x2d")),
                    basin_ids=self.basin_ids,
                    add_self_loops=bool(graph_spec.get("add_self_loops", graph_options.get("add_self_loops", False))),
                    directed=graph_spec.get("directed", graph_options.get("directed")),
                    add_reverse_edges=graph_spec.get("add_reverse_edges", graph_options.get("add_reverse_edges")),
                    show_progress=self.show_progress,
                )
                if cache_path:
                    cache_path = Path(cache_path)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(raw_graph, cache_path)
            if artifact_netcdf_path:
                export_routing_graph_netcdf(raw_graph, artifact_netcdf_path)

        self.routing_graph = normalize_routing_graph_payload(
            raw_graph,
            dtype=self.dtype_torch,
            grid_shape=self.x_info_global.get("grid_shape"),
        )
        if isinstance(self.routing_graph, dict):
            self.x_info_global["routing_graph_keys"] = list(self.routing_graph.keys())
            metadata = dict(self.routing_graph.get("metadata", {}) or {})
            self.x_info_global["routing_graph_metadata"] = metadata
            logger.info(
                "Routing graph ready | nodes=%s | edges=%s | gauges=%s",
                self.routing_graph.get("num_nodes", 0),
                self.routing_graph.get("num_edges", 0),
                self.routing_graph.get("num_gauges", 0),
            )
        self._configure_routing_domain_compaction(graph_spec)

    def _compact_spatial_array_with_index(
        self,
        values: np.ndarray,
        active_index: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(values)
        if active_index.size == 0:
            raise ValueError("Active Routing domain index cannot be empty")
        full_grid_shape = tuple(int(v) for v in self.x_info_global.get("grid_shape", ()))
        if len(full_grid_shape) != 2:
            raise ValueError("grid_shape metadata is required before compacting Routing inputs")

        if array.ndim >= 2 and tuple(array.shape[-2:]) == full_grid_shape:
            flattened = array.reshape(*array.shape[:-2], -1)
            return np.take(flattened, active_index, axis=-1)
        if array.ndim >= 2 and tuple(array.shape[:2]) == full_grid_shape:
            flattened = array.reshape(int(np.prod(full_grid_shape)), *array.shape[2:])
            return flattened[active_index]
        if array.ndim >= 1 and array.shape[-1] == int(active_index.size):
            return array
        return array

    def _compact_routing_array(self, values: np.ndarray) -> np.ndarray:
        if not self.routing_compact_domain or self.routing_active_index is None:
            return values
        return self._compact_spatial_array_with_index(values, self.routing_active_index)

    def _materialize_dynamic_dataarray(
        self,
        da: xr.DataArray,
        *,
        time_dim: str,
        y_dim: str,
        x_dim: str,
        indexer: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            self.routing_compact_domain
            and self.routing_active_y_index is not None
            and self.routing_active_x_index is not None
            and y_dim in da.dims
            and x_dim in da.dims
        ):
            site_dim = "routing_site"
            y_indexer = xr.DataArray(self.routing_active_y_index, dims=(site_dim,))
            x_indexer = xr.DataArray(self.routing_active_x_index, dims=(site_dim,))
            da = da.isel({y_dim: y_indexer, x_dim: x_indexer}).transpose(time_dim, site_dim)
            values = da.to_numpy()
        else:
            values = da.transpose(time_dim, y_dim, x_dim).to_numpy()

        if indexer is not None:
            values = values[indexer]
        return values.astype(self.dtype_np, copy=False)

    def _configure_routing_domain_compaction(self, graph_spec: Mapping[str, Any]) -> None:
        self.routing_compact_domain = bool(graph_spec.get("compact_routing_domain", False))
        self.routing_active_index = None
        self.routing_active_y_index = None
        self.routing_active_x_index = None
        if not self.routing_compact_domain:
            return
        if not isinstance(self.routing_graph, Mapping):
            raise ValueError(
                "routing.graph.compact_routing_domain=true requires a routing graph payload"
            )

        if "runoff_source_flat_index" in self.routing_graph:
            active_index = torch.as_tensor(
                self.routing_graph["runoff_source_flat_index"],
                dtype=torch.long,
            ).cpu().numpy()
        elif "flat_index" in self.routing_graph:
            active_index = torch.as_tensor(self.routing_graph["flat_index"], dtype=torch.long).cpu().numpy()
        else:
            raise ValueError(
                "routing.graph.compact_routing_domain=true requires either "
                "'runoff_source_flat_index' or 'flat_index' in the routing graph"
            )
        self.routing_active_index = np.asarray(active_index, dtype=np.int64).reshape(-1)
        grid_shape = tuple(int(v) for v in self.x_info_global.get("grid_shape", ()))
        if len(grid_shape) != 2:
            raise ValueError("grid_shape metadata is required before compacting Routing inputs")
        self.routing_active_y_index = self.routing_active_index // int(grid_shape[1])
        self.routing_active_x_index = self.routing_active_index % int(grid_shape[1])
        self.static_np = self._compact_routing_array(self.static_np)

        for group in self.optional_static_groups.values():
            group["array"] = self._compact_routing_array(np.asarray(group["array"]))
            if "field_arrays" in group:
                group["field_arrays"] = {
                    name: self._compact_routing_array(np.asarray(values))
                    for name, values in group["field_arrays"].items()
                }

        self.x_info_global["routing_compact_domain"] = True
        self.x_info_global["routing_active_flat_index"] = self.routing_active_index.copy()
        self.x_info_global["routing_active_count"] = int(self.routing_active_index.size)
        logger.info(
            "Routing domain compaction enabled | active_pixels=%s | full_pixels=%s",
            int(self.routing_active_index.size),
            int(np.prod(self.x_info_global.get("grid_shape", self.static_np.shape[-2:]))),
        )

    def _routing_field_sources(self) -> Dict[str, Dict[str, np.ndarray]]:
        sources: Dict[str, Dict[str, np.ndarray]] = {
            "static": dict(self.static_field_arrays),
            "x_static_physical": dict(self.static_field_arrays),
        }
        for group_name, group in self.optional_static_groups.items():
            if "field_arrays" in group:
                sources[group_name] = dict(group["field_arrays"])
        if "x_routing_static" in sources:
            sources["routing_static"] = dict(sources["x_routing_static"])
        return sources

    def _resolve_routing_array_spec(
        self,
        spec: Mapping[str, Any] | str | None,
        *,
        required: bool,
    ) -> np.ndarray | None:
        if spec is None:
            if required:
                raise ValueError("A routing graph array specification is required")
            return None
        if isinstance(spec, str):
            spec = {"variable": spec}
        if not isinstance(spec, Mapping):
            raise ValueError("Routing graph array specification must be a mapping or variable name")

        variable = spec.get("variable")
        if not variable:
            if required:
                raise ValueError("Routing graph array specification requires a 'variable'")
            return None

        source_name = spec.get("source")
        sources = self._routing_field_sources()
        search_order = [source_name] if source_name else ["routing_static", "x_routing_static", "static", "x_static_physical"]

        array = None
        for candidate in search_order:
            if not candidate:
                continue
            group = sources.get(str(candidate))
            if group and variable in group:
                array = group[variable]
                break
        if array is None:
            if required:
                raise KeyError(
                    f"Routing graph variable '{variable}' was not found in sources {search_order}"
                )
            return None

        array = np.asarray(array)
        if array.ndim == 2:
            return array

        channel = spec.get("channel")
        if array.ndim == 3:
            if channel is None:
                if array.shape[-1] == 1:
                    return array[..., 0]
                raise ValueError(
                    f"Routing graph variable '{variable}' has shape {tuple(array.shape)}; "
                    "specify 'channel' to select a 2-D slice"
                )
            return array[..., int(channel)]

        raise ValueError(
            f"Routing graph variable '{variable}' must resolve to 2-D or 3-D data, got {tuple(array.shape)}"
        )

    def _resolve_routing_node_feature_spec(
        self,
        spec: Mapping[str, Any] | Sequence[Any] | str | None,
        *,
        required: bool,
    ) -> np.ndarray | None:
        if spec is None:
            if required:
                raise ValueError("routing.graph.node_features is required")
            return None

        if isinstance(spec, (str, Mapping)):
            feature_array = self._resolve_routing_array_spec(spec, required=required)
            if feature_array is None:
                return None
            array = np.asarray(feature_array)
            return array if array.ndim == 3 else array[..., np.newaxis]

        if not isinstance(spec, Sequence):
            raise ValueError("routing.graph.node_features must be a mapping, string, or sequence of them")

        features: List[np.ndarray] = []
        for item in spec:
            feature_array = self._resolve_routing_array_spec(item, required=required)
            if feature_array is None:
                continue
            array = np.asarray(feature_array)
            if array.ndim == 2:
                features.append(array[..., np.newaxis])
            elif array.ndim == 3:
                features.append(array)
            else:
                raise ValueError(
                    "Each routing.graph.node_features item must resolve to [Y,X] or [Y,X,C], "
                    f"got {tuple(array.shape)}"
                )

        if not features:
            return None
        return np.concatenate(features, axis=-1)

    def _get_nested_optional_spec(self, section_name: str, subsection_name: str) -> Dict[str, Any]:
        section = self.config.section(section_name)
        spec = section.get(subsection_name, {})
        if spec is None:
            return {}
        if not isinstance(spec, dict):
            raise ValueError(f"'{section_name}.{subsection_name}' must be a mapping when provided.")
        return dict(spec)

    def _resolved_spec(self, spec: Mapping[str, Any], *, source_kind: str) -> Dict[str, Any]:
        resolved = {}
        if source_kind == "forcing":
            resolved.update(self.config.forcing)
            resolved.setdefault("file_mode", self.config.forcing_file_mode)
            resolved.setdefault("open_kwargs", self.config.forcing_open_kwargs)
            resolved.setdefault("time_dim", self.config.forcing_time_dim)
            resolved.setdefault("y_dim", self.config.forcing_y_dim)
            resolved.setdefault("x_dim", self.config.forcing_x_dim)
        elif source_kind == "static":
            resolved.update(self.config.static)
            resolved.setdefault("file_mode", self.config.static_file_mode)
            resolved.setdefault("open_kwargs", self.config.static_open_kwargs)
        resolved.update(dict(spec))
        return resolved

    def _load_static_group_from_spec(
        self,
        spec: Mapping[str, Any],
        *,
        group_name: str,
        normalize: bool,
        default_y_dim: str = "y",
        default_x_dim: str = "x",
    ) -> Dict[str, Any]:
        resolved = self._resolved_spec(spec, source_kind=str(spec.get("source", "static")))
        ds = open_dataset_from_mode(
            mode=str(resolved.get("file_mode", "auto")),
            variables=list(resolved.get("variables", [])),
            directory=resolved.get("dir"),
            file_path=resolved.get("file_path"),
            var_to_file_map=resolved.get("var_to_file_map"),
            open_kwargs=resolved.get("open_kwargs", {}),
            dataset_kind="static",
        )

        channel_arrays: List[np.ndarray] = []
        channel_names: List[str] = []
        field_arrays: Dict[str, np.ndarray] = {}
        channel_map: Dict[str, List[str]] = {}

        lat2d = np.asarray(ds.coords["lat"].values) if "lat" in ds.coords else None
        lon2d = np.asarray(ds.coords["lon"].values) if "lon" in ds.coords else None
        if lat2d is None and "LAT2D" in ds:
            lat2d = np.asarray(ds["LAT2D"].values)
        if lon2d is None and "LON2D" in ds:
            lon2d = np.asarray(ds["LON2D"].values)
        coord_y2d = None
        coord_x2d = None

        static_vars = list(resolved.get("variables", []))
        for var in self._progress(
            static_vars,
            desc=f"{group_name} vars",
            total=len(static_vars),
        ):
            if var not in ds:
                raise KeyError(f"Static variable '{var}' not found for group '{group_name}'")
            channel_array, names, field_array = _expand_static_variable(
                var,
                ds[var],
                dtype=self.dtype_np,
                y_dim=str(resolved.get("y_dim", default_y_dim)) if resolved.get("y_dim") else default_y_dim,
                x_dim=str(resolved.get("x_dim", default_x_dim)) if resolved.get("x_dim") else default_x_dim,
            )
            channel_arrays.append(channel_array)
            channel_names.extend(names)
            field_arrays[var] = field_array
            channel_map[var] = names

            if lat2d is None or lon2d is None:
                da = ds[var]
                y_dim = da.dims[-2]
                x_dim = da.dims[-1]
                if y_dim in da.coords and x_dim in da.coords:
                    yy, xx = np.meshgrid(da.coords[y_dim].values, da.coords[x_dim].values, indexing="ij")
                    if coord_y2d is None or coord_x2d is None:
                        coord_y2d = yy.astype(np.float32)
                        coord_x2d = xx.astype(np.float32)
                    if lat2d is None or lon2d is None:
                        lat2d = yy.astype(np.float32)
                        lon2d = xx.astype(np.float32)

            if coord_y2d is None or coord_x2d is None:
                da = ds[var]
                y_dim = da.dims[-2]
                x_dim = da.dims[-1]
                if y_dim in da.coords and x_dim in da.coords:
                    yy, xx = np.meshgrid(da.coords[y_dim].values, da.coords[x_dim].values, indexing="ij")
                    coord_y2d = yy.astype(np.float32)
                    coord_x2d = xx.astype(np.float32)

        ds.close()

        if not channel_arrays:
            raise RuntimeError(f"No static variables were loaded for group '{group_name}'")

        array = np.concatenate(channel_arrays, axis=0).astype(self.dtype_np, copy=False)
        if normalize:
            categorical_static = {"VEG2D", "BOTSOIL2D", "landuse", "soiltype"}
            categorical_used = categorical_static.intersection(set(static_vars))
            if categorical_used:
                logger.warning(
                    "Normalizing static group '%s' that includes categorical variables %s. "
                    "For ML runoff models, prefer excluding them initially or using one-hot/embedding handling.",
                    group_name,
                    sorted(categorical_used),
                )
            array = self._normalize_static_array(array, group_name, channel_names)

        return {
            "array": array,
            "names": channel_names,
            "field_arrays": field_arrays,
            "channel_map": channel_map,
            "lat2d": lat2d.astype(np.float32) if lat2d is not None else None,
            "lon2d": lon2d.astype(np.float32) if lon2d is not None else None,
            "y2d": coord_y2d.astype(np.float32) if coord_y2d is not None else None,
            "x2d": coord_x2d.astype(np.float32) if coord_x2d is not None else None,
        }

    def _default_daily_aggregation(self, variable: str) -> Dict[str, Any]:
        if str(variable).upper() == "RAINRATE":
            return {
                "op": "sum",
                "factor": 3600.0,
                "output_name": f"{variable}_daily",
            }
        return {"op": "mean", "factor": 1.0, "output_name": variable}

    def _build_daily_aggregation_plan(
        self,
        spec: Mapping[str, Any],
        group_names: Sequence[str],
    ) -> tuple[list[Dict[str, Any]], list[str]]:
        aggregation_spec = dict(spec.get("aggregations", spec.get("daily_aggregations", {})) or {})
        plan: list[Dict[str, Any]] = []
        output_names: list[str] = []

        for variable in group_names:
            entries = _as_list(aggregation_spec.get(variable, self._default_daily_aggregation(str(variable))))
            for raw_entry in entries:
                if isinstance(raw_entry, str):
                    entry = {"op": raw_entry}
                elif isinstance(raw_entry, Mapping):
                    entry = dict(raw_entry)
                else:
                    raise ValueError(
                        f"Daily aggregation for variable '{variable}' must be a string, mapping, or list of them"
                    )
                default_entry = self._default_daily_aggregation(str(variable))
                op = str(entry.get("op", default_entry["op"])).lower()
                if op not in {"sum", "mean", "min", "max"}:
                    raise ValueError(
                        f"Unsupported daily aggregation op '{op}' for variable '{variable}'. "
                        "Supported ops are: sum, mean, min, max"
                    )
                factor = float(entry.get("factor", default_entry.get("factor", 1.0)))
                output_name = str(
                    entry.get(
                        "output_name",
                        entry.get("name", default_entry.get("output_name", f"{variable}_{op}")),
                    )
                )
                plan.append(
                    {
                        "source_name": str(variable),
                        "source_index": int(group_names.index(variable)),
                        "op": op,
                        "factor": factor,
                        "output_name": output_name,
                    }
                )
                output_names.append(output_name)

        if not plan:
            raise ValueError("Daily ML forcing aggregation produced no output channels")
        return plan, output_names

    def _aggregate_daily_window(
        self,
        window: np.ndarray,
        time_index: pd.DatetimeIndex,
        group: Mapping[str, Any],
    ) -> np.ndarray:
        if len(time_index) != int(window.shape[0]):
            raise ValueError(
                f"Daily aggregation time length mismatch: {len(time_index)} timestamps for window {tuple(window.shape)}"
            )

        days = pd.DatetimeIndex(time_index.normalize().unique())
        min_hours = int(group.get("min_steps_per_day", group.get("min_hours_per_day", 24)))
        allow_incomplete = bool(group.get("allow_incomplete_day", False))
        daily_arrays: list[np.ndarray] = []

        normalized = time_index.normalize()
        for day in days:
            day_mask = np.asarray(normalized == day)
            if int(day_mask.sum()) < min_hours and not allow_incomplete:
                raise RuntimeError(
                    f"Daily aggregation for group has only {int(day_mask.sum())} timesteps on {day.date()}, "
                    f"but min_hours_per_day={min_hours}. Set allow_incomplete_day=true to keep partial days."
                )
            day_window = window[day_mask]
            channel_arrays: list[np.ndarray] = []
            for entry in group["aggregation_plan"]:
                values = day_window[:, int(entry["source_index"])] * float(entry["factor"])
                op = str(entry["op"])
                if op == "sum":
                    channel = np.nansum(values, axis=0)
                elif op == "mean":
                    channel = np.nanmean(values, axis=0)
                elif op == "min":
                    channel = np.nanmin(values, axis=0)
                elif op == "max":
                    channel = np.nanmax(values, axis=0)
                else:
                    raise ValueError(f"Unsupported daily aggregation op '{op}'")
                channel_arrays.append(channel.astype(self.dtype_np, copy=False))
            daily_arrays.append(np.stack(channel_arrays, axis=0))

        if not daily_arrays:
            raise RuntimeError("Daily aggregation produced an empty window")
        return np.stack(daily_arrays, axis=0).astype(self.dtype_np, copy=False)

    def _aggregate_daily_base_forcing_window(
        self,
        start: int,
        end: int,
        group: Mapping[str, Any],
    ) -> np.ndarray:
        time_slice = pd.DatetimeIndex(self.time_index[start:end])
        if time_slice.empty:
            raise RuntimeError("Requested an empty daily forcing aggregation window")

        normalized = time_slice.normalize()
        daily_arrays: list[np.ndarray] = []
        for day in pd.DatetimeIndex(normalized.unique()):
            relative = np.where(np.asarray(normalized == day))[0]
            if relative.size == 0:
                continue
            day_start = start + int(relative[0])
            day_end = start + int(relative[-1]) + 1
            window = self._get_forcing_window(day_start, day_end)[:, group["channel_indices"]]
            aggregated = self._aggregate_daily_window(
                window,
                pd.DatetimeIndex(self.time_index[day_start:day_end]),
                group,
            )
            daily_arrays.append(aggregated[0])

        if not daily_arrays:
            raise RuntimeError("Daily base forcing aggregation produced no days")
        return np.stack(daily_arrays, axis=0).astype(self.dtype_np, copy=False)

    def _load_dynamic_group_from_spec(
        self,
        spec: Mapping[str, Any],
        *,
        group_name: str,
        normalize: bool,
    ) -> Dict[str, Any]:
        source_kind = str(spec.get("source", "forcing"))
        resolved = self._resolved_spec(spec, source_kind=source_kind)
        group_names = list(resolved.get("variables", []))
        target_frequency = str(
            spec.get("target_frequency", spec.get("frequency", spec.get("resample", "")))
        ).lower()
        transforms = dict(spec.get("transforms", {}) or {})
        time_features = _normalize_time_feature_names(
            spec.get("time_features", spec.get("calendar_features", []))
        )

        if source_kind == "forcing" and all(name in self.forcing_name_to_index for name in group_names):
            if target_frequency in {"daily", "1d", "day"}:
                already_daily = bool(
                    spec.get(
                        "already_daily",
                        spec.get("native_daily", str(spec.get("source_frequency", "")).lower() in {"daily", "1d", "day"}),
                    )
                )
                if already_daily:
                    means, stds = self._dynamic_scaler_for_base_forcing(group_name, group_names) if normalize else (None, None)
                    return {
                        "kind": "base_forcing_view",
                        "names": list(group_names) + list(time_features),
                        "channel_indices": [self.forcing_name_to_index[name] for name in group_names],
                        "means": means,
                        "stds": stds,
                        "transforms": transforms,
                        "time_features": time_features,
                        "frequency": "daily",
                        "include_spinup": bool(spec.get("include_spinup", False)),
                    }

                if normalize and not (spec.get("means") and spec.get("stds")):
                    logger.warning(
                        "Ignoring normalize=true for daily ML forcing group '%s' because daily aggregation "
                        "changes channel statistics. Provide explicit means/stds or use runoff_model.input_norm.",
                        group_name,
                    )
                means = np.asarray(spec.get("means"), dtype=self.dtype_np) if spec.get("means") is not None else None
                stds = np.asarray(spec.get("stds"), dtype=self.dtype_np) if spec.get("stds") is not None else None
                aggregation_plan, output_names = self._build_daily_aggregation_plan(spec, group_names)
                names = list(output_names) + list(time_features)
                return {
                    "kind": "base_forcing_daily",
                    "names": names,
                    "data_names": output_names,
                    "source_names": group_names,
                    "channel_indices": [self.forcing_name_to_index[name] for name in group_names],
                    "aggregation_plan": aggregation_plan,
                    "frequency": "daily",
                    "include_spinup": bool(spec.get("include_spinup", False)),
                    "min_hours_per_day": int(spec.get("min_hours_per_day", 24)),
                    "allow_incomplete_day": bool(spec.get("allow_incomplete_day", False)),
                    "transforms": transforms,
                    "time_features": time_features,
                    "means": means,
                    "stds": stds,
                }

            means, stds = self._dynamic_scaler_for_base_forcing(group_name, group_names) if normalize else (None, None)
            return {
                "kind": "base_forcing_view",
                "names": group_names,
                "channel_indices": [self.forcing_name_to_index[name] for name in group_names],
                "means": means,
                "stds": stds,
                "transforms": transforms,
                "time_features": time_features,
                "frequency": "native",
                "include_spinup": True,
            }

        if source_kind == "forcing" and target_frequency in {"daily", "1d", "day"}:
            ds = open_dataset_from_mode(
                mode=str(resolved.get("file_mode", "auto")),
                variables=group_names,
                directory=resolved.get("dir"),
                file_path=resolved.get("file_path"),
                var_to_file_map=resolved.get("var_to_file_map"),
                glob_pattern=str(resolved.get("glob", "*.nc")),
                open_kwargs=resolved.get("open_kwargs", {}),
                dataset_kind="forcing",
                period_start=self.period_start,
                period_end=self.period_end,
            )
            time_dim = infer_time_coord(ds, resolved.get("time_dim"))
            y_dim = str(resolved.get("y_dim", self.config.forcing_y_dim))
            x_dim = str(resolved.get("x_dim", self.config.forcing_x_dim))
            daily_time_index = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].to_numpy()))
            available_days = pd.DatetimeIndex(daily_time_index.normalize())
            requested_days = pd.DatetimeIndex(self.time_index.normalize().unique())
            indexer = available_days.get_indexer(requested_days)
            if np.any(indexer < 0):
                missing = requested_days[indexer < 0][:5]
                ds.close()
                raise RuntimeError(
                    f"Daily dynamic group '{group_name}' is missing days aligned with Routing forcing. "
                    f"First missing examples: {list(missing)}"
                )
            if normalize:
                if spec.get("means") is not None and spec.get("stds") is not None:
                    means = np.asarray(spec.get("means"), dtype=self.dtype_np)
                    stds = np.asarray(spec.get("stds"), dtype=self.dtype_np)
                else:
                    selected_daily_times = pd.DatetimeIndex(daily_time_index[indexer])
                    means, stds = self._dynamic_scaler_from_dataset(
                        group_name,
                        group_names,
                        ds,
                        time_dim,
                        time_index=selected_daily_times,
                        transforms=transforms,
                    )
            else:
                means, stds = (None, None)
            return {
                "kind": "dataset_daily_lazy",
                "names": list(group_names) + list(time_features),
                "data_names": group_names,
                "ds": ds,
                "time_coord_name": time_dim,
                "dims": (time_dim, y_dim, x_dim),
                "selected_time_index": daily_time_index,
                "available_day_index": available_days,
                "frequency": "daily",
                "include_spinup": bool(spec.get("include_spinup", False)),
                "transforms": transforms,
                "time_features": time_features,
                "means": means,
                "stds": stds,
            }

        if (
            source_kind == "forcing"
            and str(resolved.get("file_mode", "auto")) == "multi_file"
            and bool(resolved.get("windowed_reads", False))
        ):
            manifest = load_or_build_forcing_manifest(
                resolved.get("dir"),
                str(resolved.get("glob", "*.nc")),
                manifest_path=resolved.get("manifest_path"),
                open_kwargs=resolved.get("open_kwargs", {}),
                preferred_time_dim=resolved.get("time_dim"),
                show_progress=self.show_progress,
                progress_desc=f"{self.period}:{group_name} manifest",
            )
            manifest = filter_forcing_manifest_by_period(
                manifest,
                period_start=self.period_start,
                period_end=self.period_end,
            )
            available_times = expand_forcing_manifest_time_index(manifest)
            indexer = available_times.get_indexer(self.time_index)
            if np.any(indexer < 0):
                missing = self.time_index[indexer < 0][:5]
                raise RuntimeError(
                    f"Dynamic group '{group_name}' is missing timestamps aligned with Routing forcing. "
                    f"First missing examples: {list(missing)}"
                )

            means, stds = (
                self._dynamic_scaler_from_manifest(
                    group_name,
                    group_names,
                    manifest,
                    resolved.get("open_kwargs", {}),
                    str(resolved.get("time_dim", self.config.forcing_time_dim)),
                )
                if normalize
                else (None, None)
            )
            return {
                "kind": "dataset_windowed",
                "names": list(group_names) + list(time_features),
                "data_names": group_names,
                "manifest": manifest.reset_index(drop=True),
                "open_kwargs": dict(resolved.get("open_kwargs", {}) or {}),
                "time_coord_name": str(resolved.get("time_dim", self.config.forcing_time_dim)),
                "dims": (
                    str(resolved.get("time_dim", self.config.forcing_time_dim)),
                    str(resolved.get("y_dim", self.config.forcing_y_dim)),
                    str(resolved.get("x_dim", self.config.forcing_x_dim)),
                ),
                "selected_time_index": self.time_index.copy(),
                "means": means,
                "stds": stds,
                "transforms": transforms,
                "time_features": time_features,
            }

        ds = open_dataset_from_mode(
            mode=str(resolved.get("file_mode", "auto")),
            variables=group_names,
            directory=resolved.get("dir"),
            file_path=resolved.get("file_path"),
            var_to_file_map=resolved.get("var_to_file_map"),
            glob_pattern=str(resolved.get("glob", "*.nc")),
            open_kwargs=resolved.get("open_kwargs", {}),
            dataset_kind="forcing",
            period_start=self.period_start,
            period_end=self.period_end,
        )

        time_dim = infer_time_coord(ds, resolved.get("time_dim"))
        y_dim = str(resolved.get("y_dim", self.config.forcing_y_dim))
        x_dim = str(resolved.get("x_dim", self.config.forcing_x_dim))
        time_index = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].to_numpy()))
        indexer = time_index.get_indexer(self.time_index)
        if np.any(indexer < 0):
            missing = self.time_index[indexer < 0][:5]
            raise RuntimeError(
                f"Dynamic group '{group_name}' is missing timestamps aligned with Routing forcing. "
                f"First missing examples: {list(missing)}"
            )

        n_time = len(self.time_index)
        n_y = int(ds.sizes[y_dim])
        n_x = int(ds.sizes[x_dim])
        n_vars = len(group_names)
        estimated = estimate_array_bytes((n_time, n_vars, n_y, n_x), self.dtype_np)
        available = get_available_ram_bytes()
        auto_choice = decide_io_mode(estimated, available, self.config.memory_safety_factor)
        io_mode = auto_choice if self.config.io_mode == "auto" else self.config.io_mode

        if io_mode == "preload":
            arrays = []
            for name in self._progress(
                group_names,
                desc=f"{group_name} preload vars",
                total=len(group_names),
            ):
                da = ds[name].transpose(time_dim, y_dim, x_dim)
                arrays.append(da.to_numpy()[indexer].astype(self.dtype_np, copy=False))
            array = np.stack(arrays, axis=1)
            array = self._transform_dynamic_array(array, group_names, transforms)
            means, stds = self._dynamic_scaler_from_array(group_name, group_names, array) if normalize else (None, None)
            ds.close()
            return {
                "kind": "dataset_preload",
                "names": list(group_names) + list(time_features),
                "data_names": group_names,
                "array": self._normalize_dynamic_array(array, means, stds) if normalize else array,
                "time_features": time_features,
                "means": means,
                "stds": stds,
            }

        means, stds = self._dynamic_scaler_from_dataset(group_name, group_names, ds, time_dim) if normalize else (None, None)
        return {
            "kind": "dataset_lazy",
            "names": list(group_names) + list(time_features),
            "data_names": group_names,
            "ds": ds,
            "time_coord_name": time_dim,
            "dims": (time_dim, y_dim, x_dim),
            "selected_time_index": self.time_index.copy(),
            "time_features": time_features,
            "means": means,
            "stds": stds,
        }

    def _dynamic_scaler_for_base_forcing(self, group_name: str, group_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._group_scaler_available(group_name, group_names):
            return self._load_group_scaler(group_name, group_names)
        if self.period == "train":
            if self.forcing_np is not None:
                data = self.forcing_np[:, [self.forcing_name_to_index[name] for name in group_names]]
                return self._dynamic_scaler_from_array(group_name, group_names, data)
            if self.forcing_manifest is not None:
                return self._dynamic_scaler_from_manifest(
                    group_name,
                    group_names,
                    self.forcing_manifest,
                    self.config.forcing_open_kwargs,
                    self.time_coord_name,
                )
            means = []
            stds = []
            for name in self._progress(
                group_names,
                desc=f"{group_name} scaler vars",
                total=len(group_names),
            ):
                da = self.forcing_ds[name]
                means.append(float(da.mean(skipna=True).compute().item()))
                stds.append(float(max(da.std(skipna=True).compute().item(), 1e-6)))
            stats = self.scaler.setdefault("group_stats", {})
            stats[group_name] = {
                "means": {name: mean for name, mean in zip(group_names, means)},
                "stds": {name: std for name, std in zip(group_names, stds)},
            }
            return np.asarray(means, dtype=self.dtype_np), np.asarray(stds, dtype=self.dtype_np)

        return self._load_group_scaler(group_name, group_names)

    def _dynamic_scaler_from_array(
        self,
        group_name: str,
        group_names: Sequence[str],
        array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._group_scaler_available(group_name, group_names):
            return self._load_group_scaler(group_name, group_names)
        if self.period == "train":
            means = np.asarray([float(np.nanmean(array[:, idx])) for idx in range(array.shape[1])], dtype=self.dtype_np)
            stds = np.asarray([float(max(np.nanstd(array[:, idx]), 1e-6)) for idx in range(array.shape[1])], dtype=self.dtype_np)
            stats = self.scaler.setdefault("group_stats", {})
            stats[group_name] = {
                "means": {name: float(means[idx]) for idx, name in enumerate(group_names)},
                "stds": {name: float(stds[idx]) for idx, name in enumerate(group_names)},
            }
            return means, stds
        return self._load_group_scaler(group_name, group_names)

    def _dynamic_scaler_from_dataset(
        self,
        group_name: str,
        group_names: Sequence[str],
        ds: xr.Dataset,
        time_dim: str,
        *,
        time_index: pd.DatetimeIndex | None = None,
        transforms: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._group_scaler_available(group_name, group_names):
            return self._load_group_scaler(group_name, group_names)
        if self.period == "train":
            selected_time_index = pd.DatetimeIndex(time_index) if time_index is not None else self.time_index
            means = []
            stds = []
            for name in self._progress(
                group_names,
                desc=f"{group_name} dataset scaler vars",
                total=len(group_names),
            ):
                da = ds[name].sel({time_dim: selected_time_index})
                da = self._transform_dynamic_dataarray(da, name=str(name), transforms=transforms)
                means.append(float(da.mean(skipna=True).compute().item()))
                stds.append(float(max(da.std(skipna=True).compute().item(), 1e-6)))
            stats = self.scaler.setdefault("group_stats", {})
            stats[group_name] = {
                "means": {name: mean for name, mean in zip(group_names, means)},
                "stds": {name: std for name, std in zip(group_names, stds)},
            }
            return np.asarray(means, dtype=self.dtype_np), np.asarray(stds, dtype=self.dtype_np)
        return self._load_group_scaler(group_name, group_names)

    def _dynamic_scaler_from_manifest(
        self,
        group_name: str,
        group_names: Sequence[str],
        manifest: pd.DataFrame,
        open_kwargs: Mapping[str, Any],
        time_dim: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._group_scaler_available(group_name, group_names):
            return self._load_group_scaler(group_name, group_names)
        if self.period != "train":
            return self._load_group_scaler(group_name, group_names)

        sums = np.zeros(len(group_names), dtype=np.float64)
        sums_sq = np.zeros(len(group_names), dtype=np.float64)
        counts = np.zeros(len(group_names), dtype=np.float64)

        for row in self._progress(
            manifest.itertuples(index=False),
            desc=f"{group_name} manifest scaler files",
            total=len(manifest),
        ):
            file_start = pd.Timestamp(row.start_time)
            file_end = pd.Timestamp(row.end_time)
            selected_times = self.time_index[(self.time_index >= file_start) & (self.time_index <= file_end)]
            if selected_times.empty:
                continue

            ds = open_forcing_dataset_from_files(
                [Path(str(row.path))],
                variables=group_names,
                open_kwargs=dict(open_kwargs or {}),
                preferred_time_dim=time_dim,
            )
            try:
                file_time_dim = infer_time_coord(ds, time_dim)
                file_index = pd.DatetimeIndex(pd.to_datetime(ds[file_time_dim].to_numpy()))
                indexer = file_index.get_indexer(selected_times)
                if np.any(indexer < 0):
                    missing = selected_times[indexer < 0][:5]
                    raise RuntimeError(
                        f"Manifest-backed scaler load for group '{group_name}' is missing timestamps. "
                        f"Examples: {list(missing)}"
                    )
                for idx, name in enumerate(group_names):
                    values = ds[name].to_numpy()[indexer]
                    valid = np.isfinite(values)
                    if not np.any(valid):
                        continue
                    values = values[valid].astype(np.float64, copy=False)
                    counts[idx] += float(values.size)
                    sums[idx] += float(values.sum())
                    sums_sq[idx] += float(np.square(values).sum())
            finally:
                ds.close()

        means = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts > 0,
        )
        variances = np.divide(
            sums_sq,
            counts,
            out=np.zeros_like(sums_sq),
            where=counts > 0,
        ) - np.square(means)
        stds = np.sqrt(np.maximum(variances, 0.0))
        stds = np.maximum(stds, 1.0e-6)

        stats = self.scaler.setdefault("group_stats", {})
        stats[group_name] = {
            "means": {name: float(means[idx]) for idx, name in enumerate(group_names)},
            "stds": {name: float(stds[idx]) for idx, name in enumerate(group_names)},
        }
        return np.asarray(means, dtype=self.dtype_np), np.asarray(stds, dtype=self.dtype_np)

    def _normalize_dynamic_array(
        self,
        array: np.ndarray,
        means: Optional[np.ndarray],
        stds: Optional[np.ndarray],
    ) -> np.ndarray:
        if means is None or stds is None:
            return array
        broadcast_shape = (1, int(len(means))) + (1,) * max(array.ndim - 2, 0)
        return (array - means.reshape(broadcast_shape)) / stds.reshape(broadcast_shape)

    def _transform_dynamic_array(
        self,
        array: np.ndarray,
        group_names: Sequence[str],
        transforms: Mapping[str, Any] | None,
    ) -> np.ndarray:
        if not transforms:
            return array
        result = array.astype(self.dtype_np, copy=True)
        for idx, name in enumerate(group_names):
            transform = str(transforms.get(name, transforms.get(str(name).lower(), "identity"))).lower()
            if transform in {"", "none", "identity"}:
                continue
            if transform == "log1p":
                result[:, idx] = np.log1p(np.clip(result[:, idx], a_min=0.0, a_max=None))
                continue
            raise ValueError(f"Unsupported dynamic transform '{transform}' for variable '{name}'")
        return result

    def _transform_dynamic_dataarray(
        self,
        da: xr.DataArray,
        *,
        name: str,
        transforms: Mapping[str, Any] | None,
    ) -> xr.DataArray:
        if not transforms:
            return da
        transform = str(transforms.get(name, transforms.get(str(name).lower(), "identity"))).lower()
        if transform in {"", "none", "identity"}:
            return da
        if transform == "log1p":
            return np.log1p(da.clip(min=0.0))
        raise ValueError(f"Unsupported dynamic transform '{transform}' for variable '{name}'")

    def _normalize_static_array(
        self,
        array: np.ndarray,
        group_name: str,
        group_names: Sequence[str],
    ) -> np.ndarray:
        if self._group_scaler_available(group_name, group_names):
            means, stds = self._load_group_scaler(group_name, group_names)
            return (array - means[:, None, None]) / stds[:, None, None]
        if self.period == "train":
            means = np.asarray([float(np.nanmean(array[idx])) for idx in range(array.shape[0])], dtype=self.dtype_np)
            stds = np.asarray([float(max(np.nanstd(array[idx]), 1e-6)) for idx in range(array.shape[0])], dtype=self.dtype_np)
            stats = self.scaler.setdefault("group_stats", {})
            stats[group_name] = {
                "means": {name: float(means[idx]) for idx, name in enumerate(group_names)},
                "stds": {name: float(stds[idx]) for idx, name in enumerate(group_names)},
            }
        else:
            means, stds = self._load_group_scaler(group_name, group_names)
            return (array - means[:, None, None]) / stds[:, None, None]

        return (array - means[:, None, None]) / stds[:, None, None]

    def _group_scaler_available(self, group_name: str, group_names: Sequence[str]) -> bool:
        stats = (self.scaler.get("group_stats") or {}).get(group_name)
        if not stats:
            return False
        means = stats.get("means") or {}
        stds = stats.get("stds") or {}
        return all(name in means and name in stds for name in group_names)

    def _load_group_scaler(self, group_name: str, group_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        stats = (self.scaler.get("group_stats") or {}).get(group_name)
        if not stats:
            raise ValueError(
                f"Missing scaler stats for normalized group '{group_name}'. "
                "Validation/test loaders need the training scaler."
            )
        means = np.asarray([stats["means"][name] for name in group_names], dtype=self.dtype_np)
        stds = np.asarray([stats["stds"][name] for name in group_names], dtype=self.dtype_np)
        return means, stds

    def _context_start(self, forcing_start: int, context_days: int) -> int:
        if context_days <= 0:
            return forcing_start
        context_start_time = pd.Timestamp(self.time_index[forcing_start]) - pd.DateOffset(
            days=int(context_days)
        )
        return int(self.time_index.searchsorted(context_start_time, side="left"))

    def _context_steps(self, context_start: int, forcing_start: int) -> int:
        if context_start >= forcing_start:
            return 0
        context_dates = pd.DatetimeIndex(self.time_index[context_start:forcing_start])
        return int(len(pd.DatetimeIndex(context_dates.normalize().unique())))

    def _append_lookup_window(self, forcing_start: int, forcing_end: int) -> None:
        forcing_days = pd.DatetimeIndex(self.time_index[forcing_start:forcing_end].normalize().unique())
        target_mask = self.target_time_index.normalize().isin(forcing_days)
        target_idx = np.where(target_mask)[0]

        if len(target_idx) > 0:
            if not np.isfinite(self.targets_np[target_idx[0] : target_idx[-1] + 1]).any():
                return
            self.lookup.append(
                {
                    "forcing_start": forcing_start,
                    "forcing_end": forcing_end,
                    "target_start": target_idx[0],
                    "target_end": target_idx[-1] + 1,
                }
            )

    def _build_calendar_window_lookup(self, n_time: int) -> None:
        if self.window_sequence_years is None and self.window_sequence_days is None:
            raise RuntimeError("Calendar window lookup requested without sequence_years or sequence_days.")
        if n_time <= 0:
            raise RuntimeError(f"No forcing timesteps are available for period '{self.period}'")

        available_end_exclusive = self.time_index[-1]
        if n_time > 1:
            step = self.time_index[1] - self.time_index[0]
            available_end_exclusive = available_end_exclusive + step
        else:
            available_end_exclusive = available_end_exclusive + pd.Timedelta(hours=1)

        if self.window_sequence_years is not None:
            sequence_offset = pd.DateOffset(years=int(self.window_sequence_years))
            stride_offset = pd.DateOffset(years=int(self.window_stride_years or self.window_sequence_years))
            stride_error_name = "windowing.stride_years"
        else:
            sequence_offset = pd.DateOffset(days=int(self.window_sequence_days))
            stride_offset = pd.DateOffset(days=int(self.window_stride_days or self.window_sequence_days))
            stride_error_name = "windowing.stride_days"
        current_start = pd.Timestamp(self.time_index[0])
        if self.prediction_context_days > 0:
            current_start = current_start + pd.DateOffset(days=self.prediction_context_days)

        self.lookup = []
        while current_start < available_end_exclusive:
            current_end_exclusive = current_start + sequence_offset
            if current_end_exclusive > available_end_exclusive:
                break

            forcing_start = int(self.time_index.searchsorted(current_start, side="left"))
            forcing_end = int(self.time_index.searchsorted(current_end_exclusive, side="left"))
            if forcing_end > forcing_start:
                self._append_lookup_window(forcing_start, forcing_end)

            next_start = current_start + stride_offset
            if next_start <= current_start:
                raise RuntimeError(f"Calendar window stride did not advance; check {stride_error_name}.")
            current_start = next_start

        if not self.lookup:
            raise RuntimeError(
                f"No valid calendar windows created for period '{self.period}' with "
                f"sequence_years={self.window_sequence_years}, stride_years={self.window_stride_years}, "
                f"sequence_days={self.window_sequence_days}, and stride_days={self.window_stride_days}"
            )

    def _build_lookup(self):
        n_time = len(self.time_index)

        if self.period in {"validation", "test"} and not (
            (self.window_sequence_years is not None or self.window_sequence_days is not None) and self.window_eval_periods
        ):
            forcing_start = 0
            forcing_end = n_time

            self.lookup = []
            self._append_lookup_window(forcing_start, forcing_end)
            if not self.lookup:
                raise RuntimeError(f"No target days found for period '{self.period}'")
            return

        if self.window_sequence_years is not None or self.window_sequence_days is not None:
            self._build_calendar_window_lookup(n_time)
            return

        if n_time < self.sequence_length:
            raise RuntimeError(
                f"Time length ({n_time}) is shorter than sequence_length "
                f"({self.sequence_length}) for period '{self.period}'"
            )

        self.lookup = []
        start = 0
        while start + self.sequence_length <= n_time:
            end = start + self.sequence_length
            self._append_lookup_window(start, end)
            start += self.stride

        if not self.lookup:
            raise RuntimeError(f"No valid windows created for period '{self.period}'")

    def _get_forcing_window(self, start: int, end: int) -> np.ndarray:
        if self.forcing_np is not None:
            return self.forcing_np[start:end]

        if self.forcing_manifest is not None:
            time_dim, y_dim, x_dim = self.forcing_dims
            time_slice = pd.DatetimeIndex(self.time_index[start:end])
            if time_slice.empty:
                raise RuntimeError("Requested an empty forcing window from the manifest-backed loader.")
            manifest_rows = select_forcing_manifest_rows_for_window(
                self.forcing_manifest,
                window_start=time_slice[0],
                window_end=time_slice[-1],
            )
            ds = open_forcing_dataset_from_files(
                [Path(str(path)) for path in manifest_rows["path"].tolist()],
                variables=self.forcing_names,
                open_kwargs=self.config.forcing_open_kwargs,
                preferred_time_dim=time_dim,
            )
            try:
                file_time_dim = infer_time_coord(ds, time_dim)
                file_index = pd.DatetimeIndex(pd.to_datetime(ds[file_time_dim].to_numpy()))
                indexer = file_index.get_indexer(time_slice)
                if np.any(indexer < 0):
                    missing = time_slice[indexer < 0][:5]
                    raise RuntimeError(
                        "Manifest-backed forcing window is missing requested timestamps. "
                        f"Examples: {list(missing)}"
                    )
                arrays = []
                for var in self.forcing_names:
                    arrays.append(
                        self._materialize_dynamic_dataarray(
                            ds[var],
                            time_dim=file_time_dim,
                            y_dim=y_dim,
                            x_dim=x_dim,
                            indexer=indexer,
                        )
                    )
                return np.stack(arrays, axis=1)
            finally:
                ds.close()

        time_dim, y_dim, x_dim = self.forcing_dims
        time_slice = self.time_index[start:end]
        arrays = []
        for var in self.forcing_names:
            arrays.append(
                self._materialize_dynamic_dataarray(
                    self.forcing_ds[var].sel({time_dim: time_slice}),
                    time_dim=time_dim,
                    y_dim=y_dim,
                    x_dim=x_dim,
                )
            )
        return np.stack(arrays, axis=1)

    def _get_dynamic_group_window(self, group: Mapping[str, Any], start: int, end: int) -> np.ndarray:
        kind = group["kind"]
        if kind == "base_forcing_view":
            window = self._get_forcing_window(start, end)[:, group["channel_indices"]]
            data_names = group.get("data_names", group["names"])
            window = self._transform_dynamic_array(window, data_names, group.get("transforms"))
            window = self._normalize_dynamic_array(window, group.get("means"), group.get("stds"))
            return self._append_time_features(
                window,
                pd.DatetimeIndex(self.time_index[start:end]),
                group.get("time_features", []),
            )

        if kind == "base_forcing_daily":
            daily = self._aggregate_daily_base_forcing_window(start, end, group)
            data_names = group.get("data_names", group["names"])
            daily = self._transform_dynamic_array(daily, data_names, group.get("transforms"))
            daily = self._normalize_dynamic_array(daily, group.get("means"), group.get("stds"))
            days = pd.DatetimeIndex(self.time_index[start:end].normalize().unique())
            return self._append_time_features(daily, days, group.get("time_features", []))

        if kind == "dataset_preload":
            window = group["array"][start:end]
            return self._append_time_features(
                window,
                pd.DatetimeIndex(self.time_index[start:end]),
                group.get("time_features", []),
            )

        if kind == "dataset_lazy":
            time_dim, y_dim, x_dim = group["dims"]
            time_slice = group["selected_time_index"][start:end]
            arrays = []
            data_names = group.get("data_names", group["names"])
            for var in data_names:
                arrays.append(
                    self._materialize_dynamic_dataarray(
                        group["ds"][var].sel({time_dim: time_slice}),
                        time_dim=time_dim,
                        y_dim=y_dim,
                        x_dim=x_dim,
                    )
                )
            stacked = np.stack(arrays, axis=1)
            stacked = self._transform_dynamic_array(stacked, data_names, group.get("transforms"))
            stacked = self._normalize_dynamic_array(stacked, group.get("means"), group.get("stds"))
            return self._append_time_features(stacked, pd.DatetimeIndex(time_slice), group.get("time_features", []))

        if kind == "dataset_daily_lazy":
            time_dim, y_dim, x_dim = group["dims"]
            days = pd.DatetimeIndex(self.time_index[start:end].normalize().unique())
            indexer = group["available_day_index"].get_indexer(days)
            if np.any(indexer < 0):
                missing = days[indexer < 0][:5]
                raise RuntimeError(
                    f"Daily dynamic group window is missing days aligned with forcing. Examples: {list(missing)}"
                )
            arrays = []
            data_names = group.get("data_names", group["names"])
            for var in data_names:
                arrays.append(
                    self._materialize_dynamic_dataarray(
                        group["ds"][var].isel({time_dim: indexer}),
                        time_dim=time_dim,
                        y_dim=y_dim,
                        x_dim=x_dim,
                    )
                )
            stacked = np.stack(arrays, axis=1)
            stacked = self._transform_dynamic_array(stacked, data_names, group.get("transforms"))
            stacked = self._normalize_dynamic_array(stacked, group.get("means"), group.get("stds"))
            return self._append_time_features(stacked, days, group.get("time_features", []))

        if kind == "dataset_windowed":
            time_dim, y_dim, x_dim = group["dims"]
            time_slice = pd.DatetimeIndex(group["selected_time_index"][start:end])
            manifest_rows = select_forcing_manifest_rows_for_window(
                group["manifest"],
                window_start=time_slice[0],
                window_end=time_slice[-1],
            )
            ds = open_forcing_dataset_from_files(
                [Path(str(path)) for path in manifest_rows["path"].tolist()],
                variables=group["names"],
                open_kwargs=group.get("open_kwargs", {}),
                preferred_time_dim=time_dim,
            )
            try:
                file_time_dim = infer_time_coord(ds, time_dim)
                file_index = pd.DatetimeIndex(pd.to_datetime(ds[file_time_dim].to_numpy()))
                indexer = file_index.get_indexer(time_slice)
                if np.any(indexer < 0):
                    missing = time_slice[indexer < 0][:5]
                    raise RuntimeError(
                        f"Dynamic group manifest-backed window is missing timestamps. Examples: {list(missing)}"
                )
                arrays = []
                data_names = group.get("data_names", group["names"])
                for var in data_names:
                    arrays.append(
                        self._materialize_dynamic_dataarray(
                            ds[var],
                            time_dim=file_time_dim,
                            y_dim=y_dim,
                            x_dim=x_dim,
                            indexer=indexer,
                        )
                    )
                stacked = np.stack(arrays, axis=1)
            finally:
                ds.close()
            data_names = group.get("data_names", group["names"])
            stacked = self._transform_dynamic_array(stacked, data_names, group.get("transforms"))
            stacked = self._normalize_dynamic_array(stacked, group.get("means"), group.get("stds"))
            return self._append_time_features(stacked, time_slice, group.get("time_features", []))

        raise ValueError(f"Unsupported dynamic group kind: {kind}")
