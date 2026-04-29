import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


class RoutingConfig:
    """Configuration reader for the standalone routing dataloader.

    The YAML schema is intentionally nested to make the data layout explicit.
    Required top-level keys are:
        - data_dir
        - train_basin_file, validation_basin_file, test_basin_file
        - train_start_date, train_end_date, validation_start_date, validation_end_date,
          test_start_date, test_end_date
        - forcing
        - static
        - targets
        - sequence_length

    Minimal example is provided in ``config_example_routing.yml``.
    """

    PATH_KEYS = {
        "data_dir",
        "train_basin_file",
        "validation_basin_file",
        "test_basin_file",
        "run_dir",
    }

    DATE_KEYS = {
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "test_start_date",
        "test_end_date",
    }

    def __init__(self, yml_path: Path):
        yml_path = Path(yml_path)
        if not yml_path.is_file():
            raise FileNotFoundError(f"Config file not found: {yml_path}")

        with yml_path.open("r") as fp:
            yaml = YAML(typ="safe")
            self._cfg = yaml.load(fp) or {}

        self._parse_top_level_values()
        self._validate()

    @classmethod
    def from_yaml(cls, yml_path: Path) -> "RoutingConfig":
        return cls(yml_path)

    def _parse_top_level_values(self):
        for key in list(self._cfg.keys()):
            if key in self.PATH_KEYS and self._cfg[key] is not None:
                self._cfg[key] = Path(self._cfg[key])
            elif key in self.DATE_KEYS and self._cfg[key] is not None:
                self._cfg[key] = pd.to_datetime(self._cfg[key])
            elif isinstance(self._cfg[key], dict):
                self._cfg[key] = self._coerce_nested_paths(self._cfg[key])

    def _coerce_nested_paths(self, value: Any) -> Any:
        if isinstance(value, dict):
            coerced = {}
            for key, item in value.items():
                if item is None:
                    coerced[key] = item
                elif key.endswith(("_dir", "_path", "_file")):
                    coerced[key] = Path(item)
                elif key.endswith("_files") and isinstance(item, list):
                    coerced[key] = [Path(v) for v in item]
                else:
                    coerced[key] = self._coerce_nested_paths(item)
            return coerced
        if isinstance(value, list):
            return [self._coerce_nested_paths(item) for item in value]
        return value

    def _validate(self):
        def _validate_positive_number(name: str, value: Any) -> None:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a positive number.") from exc
            if numeric_value <= 0.0:
                raise ValueError(f"{name} must be a positive number.")

        def _validate_gauge_weight_mapping(name: str, mapping: Any) -> None:
            if mapping is None:
                return
            if not isinstance(mapping, dict):
                raise ValueError(f"{name} must be a YAML mapping of gauge_id: positive_weight.")
            for gauge_id, weight in mapping.items():
                gauge_text = str(gauge_id).strip()
                if not gauge_text:
                    raise ValueError(f"{name} contains an empty gauge identifier.")
                _validate_positive_number(f"{name}[{gauge_text!r}]", weight)

        mandatory = [
            "data_dir",
            "train_basin_file",
            "validation_basin_file",
            "test_basin_file",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "test_start_date",
            "test_end_date",
            "forcing",
            "static",
            "targets",
        ]
        for key in mandatory:
            if self._cfg.get(key) is None:
                raise ValueError(f"'{key}' is mandatory in the YAML config.")

        windowing_cfg = self.section("windowing")
        has_step_windows = self._cfg.get("sequence_length") is not None
        has_calendar_windows = windowing_cfg.get("sequence_years") is not None or windowing_cfg.get("sequence_days") is not None
        if not has_step_windows and not has_calendar_windows:
            raise ValueError("Either 'sequence_length', 'windowing.sequence_years', or 'windowing.sequence_days' must be provided.")
        if windowing_cfg.get("sequence_years") is not None and windowing_cfg.get("sequence_days") is not None:
            raise ValueError("Use either windowing.sequence_years or windowing.sequence_days, not both.")
        if windowing_cfg.get("sequence_years") is not None and int(windowing_cfg.get("sequence_years")) <= 0:
            raise ValueError("windowing.sequence_years must be a positive integer.")
        if windowing_cfg.get("sequence_days") is not None and int(windowing_cfg.get("sequence_days")) <= 0:
            raise ValueError("windowing.sequence_days must be a positive integer.")
        if windowing_cfg.get("stride_years") is not None and int(windowing_cfg.get("stride_years")) <= 0:
            raise ValueError("windowing.stride_years must be a positive integer when provided.")
        if windowing_cfg.get("stride_days") is not None and int(windowing_cfg.get("stride_days")) <= 0:
            raise ValueError("windowing.stride_days must be a positive integer when provided.")
        if windowing_cfg.get("stride_years") is not None and windowing_cfg.get("stride_days") is not None:
            raise ValueError("Use either windowing.stride_years or windowing.stride_days, not both.")
        if (
            windowing_cfg.get("spinup_years") is not None
            and int(windowing_cfg.get("spinup_years")) <= 0
        ):
            raise ValueError("windowing.spinup_years must be a positive integer when provided.")
        if int(self._cfg.get("spinup_length", 0) or 0) > 0 and windowing_cfg.get("spinup_years") is not None:
            raise ValueError("Use either top-level spinup_length or windowing.spinup_years, not both.")

        for nested_name in ["forcing", "static", "targets"]:
            if not isinstance(self._cfg[nested_name], dict):
                raise ValueError(f"'{nested_name}' must be a YAML mapping.")

        if not self.forcing_variables:
            raise ValueError("forcing.variables must not be empty.")
        if not self.target_variables:
            raise ValueError("targets.variables must not be empty.")

        if self.io_mode not in {"auto", "preload", "lazy"}:
            raise ValueError("io_mode must be one of: auto, preload, lazy")

        if self.forcing_file_mode not in {"single_file", "multi_file", "multi_zarr_yearly", "per_variable", "zarr", "auto"}:
            raise ValueError(
                "forcing.file_mode must be one of: single_file, multi_file, multi_zarr_yearly, per_variable, zarr, auto"
            )

        if self.static_file_mode not in {"single_file", "per_variable", "zarr", "auto"}:
            raise ValueError("static.file_mode must be one of: single_file, per_variable, zarr, auto")

        runoff_cfg = self.section("runoff_model")
        runoff_model_type = str(runoff_cfg.get("type", "lstm")).lower()
        if runoff_cfg and runoff_model_type not in {
            "lstm",
            "spatial_lstm",
            "temporal_conv",
            "tcn",
            "spatial_temporal_conv",
            "precomputed",
            "identity",
            "passthrough",
        }:
            raise ValueError("runoff_model.type must be one of: lstm, temporal_conv, precomputed")
        execution_cfg = self.section("execution")
        if execution_cfg:
            mode = str(execution_cfg.get("mode", "unified")).lower()
            if mode not in {"unified", "split_devices"}:
                raise ValueError("execution.mode must be one of: unified, split_devices")
        runoff_warmup_keys = [key for key in ("warmup_days", "lstm_warmup_days") if runoff_cfg.get(key) is not None]
        for key in runoff_warmup_keys:
            if int(runoff_cfg.get(key)) < 0:
                raise ValueError(f"runoff_model.{key} must be non-negative when provided.")
        if len(runoff_warmup_keys) == 2 and int(runoff_cfg["warmup_days"]) != int(runoff_cfg["lstm_warmup_days"]):
            raise ValueError("Use either runoff_model.warmup_days or runoff_model.lstm_warmup_days, not both.")

        routing_cfg = self.section("routing")
        graph_cfg = routing_cfg.get("graph", {}) if isinstance(routing_cfg, dict) else {}
        if graph_cfg:
            graph_mode = str(
                graph_cfg.get(
                    "mode",
                    "load" if graph_cfg.get("file_path") else "build" if graph_cfg.get("builder") else "load",
                )
            ).lower()
            if graph_mode not in {"load", "build"}:
                raise ValueError("routing.graph.mode must be one of: load, build")
            if graph_mode == "build":
                builder = str(graph_cfg.get("builder", "grid_4_neighbor")).lower()
                if builder not in {
                    "grid_4_neighbor",
                    "grid_8_neighbor",
                    "flow_direction_d8",
                    "dem_downhill_d8",
                    "flowline_network",
                    "river_network",
                    "flowlines",
                }:
                    raise ValueError(
                        "routing.graph.builder must be one of: grid_4_neighbor, grid_8_neighbor, "
                        "flow_direction_d8, dem_downhill_d8, flowline_network"
                    )
                if builder in {"flowline_network", "river_network", "flowlines"} and not graph_cfg.get("flowlines", graph_cfg.get("flowline")):
                    raise ValueError("routing.graph.builder='flowline_network' requires routing.graph.flowlines")
                edge_features_cfg = graph_cfg.get("edge_features", {})
                if edge_features_cfg and not isinstance(edge_features_cfg, dict):
                    raise ValueError("routing.graph.edge_features must be a mapping when provided")

        routing_model_cfg = self.section("routing_model")
        if routing_model_cfg.get("routing_lag_context_days") is not None:
            if int(routing_model_cfg.get("routing_lag_context_days")) < 0:
                raise ValueError("routing_model.routing_lag_context_days must be non-negative when provided.")
        training_cfg = self.section("training")
        if training_cfg.get("weight_decay") is not None:
            weight_decay = float(training_cfg.get("weight_decay"))
            if weight_decay < 0.0:
                raise ValueError("training.weight_decay must be non-negative.")
        if training_cfg.get("mse_loss_weight") is not None:
            if float(training_cfg.get("mse_loss_weight")) < 0.0:
                raise ValueError("training.mse_loss_weight must be non-negative.")
        if training_cfg.get("kge_loss_weight") is not None:
            if float(training_cfg.get("kge_loss_weight")) < 0.0:
                raise ValueError("training.kge_loss_weight must be non-negative.")
        if training_cfg.get("peak_flow_weight") is not None:
            if float(training_cfg.get("peak_flow_weight")) < 0.0:
                raise ValueError("training.peak_flow_weight must be non-negative.")
        if training_cfg.get("peak_flow_power") is not None:
            if float(training_cfg.get("peak_flow_power")) <= 0.0:
                raise ValueError("training.peak_flow_power must be positive.")
        if training_cfg.get("peak_flow_max_weight") is not None:
            if float(training_cfg.get("peak_flow_max_weight")) <= 0.0:
                raise ValueError("training.peak_flow_max_weight must be positive.")
        if training_cfg.get("jkge_benchmark") is not None:
            benchmark = str(training_cfg.get("jkge_benchmark")).lower()
            if benchmark not in {"moving_average", "section_mean"}:
                raise ValueError(
                    "training.jkge_benchmark must be one of: moving_average, section_mean"
                )
        if training_cfg.get("jkge_window") is not None:
            if int(training_cfg.get("jkge_window")) <= 0:
                raise ValueError("training.jkge_window must be positive.")
        if training_cfg.get("jkge_section_length") is not None:
            if int(training_cfg.get("jkge_section_length")) <= 0:
                raise ValueError("training.jkge_section_length must be positive.")
        if training_cfg.get("jkge_eps") is not None:
            if float(training_cfg.get("jkge_eps")) <= 0.0:
                raise ValueError("training.jkge_eps must be positive.")
        if training_cfg.get("loss_target_space") is not None:
            loss_target_space = str(training_cfg.get("loss_target_space")).lower()
            if loss_target_space not in {"normalized", "physical", "original", "unscaled"}:
                raise ValueError(
                    "training.loss_target_space must be one of: normalized, physical, original, unscaled"
                )
        curriculum_cfg = self.section("curriculum")
        if curriculum_cfg.get("early_stopping_scope") is not None:
            scope = str(curriculum_cfg.get("early_stopping_scope")).lower()
            if scope not in {"global", "stage"}:
                raise ValueError("curriculum.early_stopping_scope must be one of: global, stage")
        if curriculum_cfg.get("stage_learning_rate_decay") is not None:
            _validate_positive_number(
                "curriculum.stage_learning_rate_decay",
                curriculum_cfg.get("stage_learning_rate_decay"),
            )
        if curriculum_cfg.get("outlet_gauge_weight") is not None:
            _validate_positive_number(
                "curriculum.outlet_gauge_weight",
                curriculum_cfg.get("outlet_gauge_weight"),
            )
        _validate_gauge_weight_mapping(
            "curriculum.gauge_weights",
            curriculum_cfg.get("gauge_weights"),
        )
        stages_cfg = curriculum_cfg.get("stages")
        if stages_cfg is not None and not isinstance(stages_cfg, list):
            raise ValueError("curriculum.stages must be a YAML list when provided.")
        for stage_index, stage in enumerate(stages_cfg or [], start=1):
            if not isinstance(stage, dict):
                raise ValueError(f"curriculum.stages[{stage_index}] must be a YAML mapping.")
            if stage.get("epochs") is not None and int(stage.get("epochs")) <= 0:
                raise ValueError(f"curriculum.stages[{stage_index}].epochs must be a positive integer.")
            if stage.get("learning_rate") is not None:
                _validate_positive_number(
                    f"curriculum.stages[{stage_index}].learning_rate",
                    stage.get("learning_rate"),
                )
            if stage.get("outlet_gauge_weight") is not None:
                _validate_positive_number(
                    f"curriculum.stages[{stage_index}].outlet_gauge_weight",
                    stage.get("outlet_gauge_weight"),
                )
            _validate_gauge_weight_mapping(
                f"curriculum.stages[{stage_index}].gauge_weights",
                stage.get("gauge_weights"),
            )

        transfer_cfg = self.section("runoff_transfer")
        if not transfer_cfg and isinstance(routing_cfg, dict):
            transfer_cfg = routing_cfg.get("runoff_transfer", {}) or {}
        if transfer_cfg:
            transfer_type = str(transfer_cfg.get("type", transfer_cfg.get("mode", "fixed"))).lower()
            if transfer_type not in {"fixed", "neural", "learned"}:
                raise ValueError("runoff_transfer.type must be one of: fixed, neural, learned")
            weight_strategy = str(transfer_cfg.get("weight_strategy", "stored")).lower()
            if weight_strategy not in {
                "stored",
                "cell_area",
                "inverse_distance",
                "exp_distance",
                "downhill",
                "downhill_distance",
            }:
                raise ValueError(
                    "runoff_transfer.weight_strategy must be one of: "
                    "stored, cell_area, inverse_distance, exp_distance, downhill, downhill_distance"
                )
            if int(transfer_cfg.get("hidden_dim", 16) or 16) <= 0:
                raise ValueError("runoff_transfer.hidden_dim must be positive when provided")

    def dump_config(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "config.yml"
        yaml = YAML()

        def _serialize(obj: Any):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, pd.Timestamp):
                return obj.strftime("%Y-%m-%d")
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            return obj

        with out.open("w") as fp:
            yaml.dump(_serialize(self._cfg), fp)

    def _get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

    def _get_nested(self, section: str, key: str, default: Any = None) -> Any:
        return (self._cfg.get(section) or {}).get(key, default)

    def section(self, section_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        section = self._cfg.get(section_name, default or {})
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise ValueError(f"'{section_name}' must be a YAML mapping when provided.")
        return dict(section)

    def set_noah_config_path(self, path: str | Path) -> None:
        """Expose an optional Noah config path to dataset-side static feature builders."""
        path_text = str(path)
        self._cfg.setdefault("noah", {})["config_path"] = path_text
        ml_static = (self._cfg.get("ml") or {}).get("static")
        if isinstance(ml_static, dict):
            source = str(ml_static.get("source", ml_static.get("kind", ml_static.get("type", "")))).lower()
            if source in {"noah_table", "noah_tables", "noah_table_priors", "table_priors", "noah_parameter_priors"}:
                ml_static["noah_config"] = path_text

    @property
    def experiment_name(self) -> str:
        return self._get("experiment_name", "routing_run")

    @property
    def run_dir(self) -> Optional[Path]:
        return self._get("run_dir")

    @property
    def device(self) -> str:
        return self._get("device", "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def data_dir(self) -> Path:
        return self._get("data_dir")

    @property
    def train_basin_file(self) -> Path:
        return self._get("train_basin_file")

    @property
    def validation_basin_file(self) -> Path:
        return self._get("validation_basin_file")

    @property
    def test_basin_file(self) -> Path:
        return self._get("test_basin_file")

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self._get("train_start_date")

    @property
    def train_end_date(self) -> pd.Timestamp:
        return self._get("train_end_date")

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return self._get("validation_start_date")

    @property
    def validation_end_date(self) -> pd.Timestamp:
        return self._get("validation_end_date")

    @property
    def test_start_date(self) -> pd.Timestamp:
        return self._get("test_start_date")

    @property
    def test_end_date(self) -> pd.Timestamp:
        return self._get("test_end_date")

    @property
    def sequence_length(self) -> int:
        return int(self._get("sequence_length", 0) or 0)

    @property
    def spinup_length(self) -> int:
        return int(self._get("spinup_length", 0))

    @property
    def stride(self) -> int:
        default_stride = self.sequence_length if self.sequence_length > 0 else 0
        return int(self._get("stride", default_stride) or default_stride)

    @property
    def windowing(self) -> Dict[str, Any]:
        return self.section("windowing")

    @property
    def window_sequence_years(self) -> Optional[int]:
        value = self.windowing.get("sequence_years")
        return None if value is None else int(value)

    @property
    def window_sequence_days(self) -> Optional[int]:
        value = self.windowing.get("sequence_days")
        return None if value is None else int(value)

    @property
    def window_stride_years(self) -> Optional[int]:
        value = self.windowing.get("stride_years")
        if value is None:
            sequence_years = self.window_sequence_years
            return None if sequence_years is None else int(sequence_years)
        return int(value)

    @property
    def window_stride_days(self) -> Optional[int]:
        value = self.windowing.get("stride_days")
        if value is None:
            sequence_days = self.window_sequence_days
            return None if sequence_days is None else int(sequence_days)
        return int(value)

    @property
    def window_spinup_years(self) -> Optional[int]:
        value = self.windowing.get("spinup_years")
        return None if value is None else int(value)

    @property
    def normalize_data(self) -> bool:
        return bool(self._get("normalize_data", False))

    @property
    def num_workers(self) -> int:
        return int(self._get("num_workers", 0))

    @property
    def batch_size(self) -> int:
        return int(self._get("batch_size", 1))

    @property
    def dtype(self) -> torch.dtype:
        dtype_name = str(self._get("dtype", "float32"))
        try:
            return getattr(torch, dtype_name)
        except AttributeError:
            logger.warning("Unknown dtype '%s'. Falling back to torch.float32.", dtype_name)
            return torch.float32

    @property
    def io_mode(self) -> str:
        return str(self._get("io_mode", "auto"))

    @property
    def memory_safety_factor(self) -> float:
        return float(self._get("memory_safety_factor", 0.65))

    @property
    def preload_static(self) -> bool:
        return bool(self._get("preload_static", True))

    @property
    def preload_targets(self) -> bool:
        return bool(self._get("preload_targets", True))

    @property
    def forcing(self) -> Dict[str, Any]:
        return self._get("forcing", {})

    @property
    def static(self) -> Dict[str, Any]:
        return self._get("static", {})

    @property
    def targets(self) -> Dict[str, Any]:
        return self._get("targets", {})

    @property
    def ml(self) -> Dict[str, Any]:
        return self.section("ml")

    @property
    def runoff_model(self) -> Dict[str, Any]:
        return self.section("runoff_model")

    @property
    def routing(self) -> Dict[str, Any]:
        return self.section("routing")

    @property
    @property
    def execution(self) -> Dict[str, Any]:
        return self.section("execution")

    @property
    def execution_mode(self) -> str:
        return str(self.execution.get("mode", "unified")).lower()

    @property
    def execution_routing_device(self) -> str:
        return str(self.execution.get("routing_device", self.device))

    @property
    def execution_parameter_model_device(self) -> str:
        return str(self.execution.get("parameter_model_device", self.device))

    @property
    def execution_routing_device(self) -> str:
        return str(self.execution.get("routing_device", self.device))

    @property
    def execution_routing_parameter_provider_device(self) -> str:
        return str(self.execution.get("routing_parameter_provider_device", self.execution_routing_device))

    @property
    def execution_runoff_cache_dir(self) -> Optional[Path]:
        return self.execution.get("runoff_cache_dir")

    @property
    def execution_reuse_runoff_cache(self) -> bool:
        return bool(self.execution.get("reuse_runoff_cache", True))

    @property
    def execution_runoff_cache_outputs(self) -> list[str]:
        return list(self.execution.get("runoff_cache_outputs", ["runoff_total"]))

    @property
    def forcing_dir(self) -> Optional[Path]:
        return self._get_nested("forcing", "dir")

    @property
    def forcing_variables(self):
        return list(self._get_nested("forcing", "variables", []))

    @property
    def forcing_file_mode(self) -> str:
        return str(self._get_nested("forcing", "file_mode", "auto"))

    @property
    def forcing_file_path(self) -> Optional[Path]:
        return self._get_nested("forcing", "file_path")

    @property
    def forcing_glob(self) -> str:
        return str(self._get_nested("forcing", "glob", "*.nc"))

    @property
    def forcing_manifest_path(self) -> Optional[Path]:
        return self._get_nested("forcing", "manifest_path")

    @property
    def forcing_windowed_reads(self) -> bool:
        return bool(self._get_nested("forcing", "windowed_reads", False))

    @property
    def forcing_time_dim(self) -> str:
        return str(self._get_nested("forcing", "time_dim", "time"))

    @property
    def forcing_y_dim(self) -> str:
        return str(self._get_nested("forcing", "y_dim", "lat"))

    @property
    def forcing_x_dim(self) -> str:
        return str(self._get_nested("forcing", "x_dim", "lon"))

    @property
    def forcing_var_to_file_map(self) -> Dict[str, Any]:
        return dict(self._get_nested("forcing", "var_to_file_map", {}) or {})

    @property
    def forcing_open_kwargs(self) -> Dict[str, Any]:
        return dict(self._get_nested("forcing", "open_kwargs", {}) or {})

    @property
    def static_dir(self) -> Optional[Path]:
        return self._get_nested("static", "dir")

    @property
    def static_variables(self):
        return list(self._get_nested("static", "variables", []))

    @property
    def static_file_mode(self) -> str:
        return str(self._get_nested("static", "file_mode", "auto"))

    @property
    def static_file_path(self) -> Optional[Path]:
        return self._get_nested("static", "file_path")

    @property
    def static_var_to_file_map(self) -> Dict[str, Any]:
        return dict(self._get_nested("static", "var_to_file_map", {}) or {})

    @property
    def static_open_kwargs(self) -> Dict[str, Any]:
        return dict(self._get_nested("static", "open_kwargs", {}) or {})

    @property
    def target_variables(self):
        return list(self._get_nested("targets", "variables", []))

    @property
    def target_dir(self) -> Optional[Path]:
        return self._get_nested("targets", "dir")

    @property
    def target_file_pattern(self) -> str:
        return str(self._get_nested("targets", "file_pattern", "{basin_id}.csv"))

    @property
    def target_date_column(self) -> str:
        return str(self._get_nested("targets", "date_column", "date"))

    @property
    def target_separator(self) -> str:
        return str(self._get_nested("targets", "separator", ","))

    @property
    def target_dtype(self) -> str:
        return str(self._get_nested("targets", "dtype", "float32"))

    @property
    def target_basin_id_column(self) -> Optional[str]:
        value = self._get_nested("targets", "basin_id_column")
        return None if value in [None, ""] else str(value)

    @property
    def target_unit_conversion(self) -> str:
        return str(self._get_nested("targets", "unit_conversion", "auto"))

    @property
    def save_scaler(self) -> bool:
        return bool(self._get("save_scaler", False))

    def __getattr__(self, name: str) -> Any:
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"RoutingConfig has no attribute '{name}'")
