from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from lstm_gnn_routing.dataset.batcher import RoutingBatcher
from lstm_gnn_routing.dataset.dataset import RoutingDataset
from lstm_gnn_routing.training.early_stopper import EarlyStopper
from lstm_gnn_routing.training.losses import get_loss_function
from lstm_gnn_routing.training.model_factory import (
    build_routing_model,
    build_runoff_model,
    build_runoff_transfer_model,
)
from lstm_gnn_routing.utils.config import RoutingConfig

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _subset_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def _move_mapping_to_device(mapping: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        result[key] = value if key == "x_info" else _to_device(value, device)
    return result


def _move_mapping_values_to_device(mapping: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else _to_device(value, device)
        for key, value in mapping.items()
    }


def _trim_prediction_context(predictions: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
    context_steps = batch.get("prediction_context_steps")
    if context_steps is None:
        context_steps = batch.get("routing_context_steps")
    if context_steps is None:
        return predictions
    context_tensor = torch.as_tensor(context_steps, device=predictions.device).reshape(-1)
    if context_tensor.numel() == 0:
        return predictions
    first_context = int(context_tensor[0].item())
    if first_context <= 0:
        return predictions
    if not torch.all(context_tensor == first_context):
        raise ValueError("Batched samples must have the same prediction_context_steps for context trimming.")
    if predictions.ndim < 2:
        raise ValueError(
            f"Cannot trim prediction context from prediction shape {tuple(predictions.shape)}; expected [B,T,...]"
        )
    if predictions.shape[1] <= first_context:
        raise ValueError(
            f"Prediction context length {first_context} is not smaller than prediction time length {predictions.shape[1]}"
        )
    return predictions[:, first_context:]


def _trim_runoff_outputs_for_routing(
    runoff_outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    trim_steps = batch.get("runoff_pre_routing_trim_steps")
    if trim_steps is None:
        return dict(runoff_outputs)

    first_tensor = next((value for value in runoff_outputs.values() if isinstance(value, torch.Tensor)), None)
    if first_tensor is None:
        return dict(runoff_outputs)

    trim_tensor = torch.as_tensor(trim_steps, device=first_tensor.device).reshape(-1)
    if trim_tensor.numel() == 0:
        return dict(runoff_outputs)
    first_trim = int(trim_tensor[0].item())
    if first_trim <= 0:
        return dict(runoff_outputs)
    if not torch.all(trim_tensor == first_trim):
        raise ValueError("Batched samples must have the same runoff_pre_routing_trim_steps.")

    trimmed: dict[str, Any] = {}
    for key, value in runoff_outputs.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            trimmed[key] = value
            continue
        if value.shape[1] <= first_trim:
            raise ValueError(
                f"Runoff warm-up trim length {first_trim} is not smaller than runoff output time length {value.shape[1]} for '{key}'."
            )
        trimmed[key] = value[:, first_trim:]
    return trimmed


def _ensure_csv_schema(path: Path, fieldnames: list[str]) -> bool:
    """Make sure an on-disk CSV has the requested header, upgrading old files if needed."""

    if not path.exists():
        return True

    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        existing_fields = list(reader.fieldnames or [])
        if existing_fields == fieldnames:
            return False
        rows = list(reader)

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    logger.info(
        "Upgraded CSV schema for %s | old_fields=%s | new_fields=%s",
        path,
        existing_fields,
        fieldnames,
    )
    return False


def _safe_checkpoint_token(value: str) -> str:
    """Return a filesystem-friendly token for checkpoint filenames."""

    result = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return result.strip("_") or "stage"


class _PerGaugeValidationMetrics:
    """Aggregate per-gauge series statistics across an entire validation epoch."""

    def __init__(self, gauge_ids: list[str] | None = None):
        self.gauge_ids = tuple(str(value) for value in (gauge_ids or ()))
        self.count: torch.Tensor | None = None
        self.sum_pred: torch.Tensor | None = None
        self.sum_obs: torch.Tensor | None = None
        self.sum_pred_sq: torch.Tensor | None = None
        self.sum_obs_sq: torch.Tensor | None = None
        self.sum_prod: torch.Tensor | None = None
        self.sum_sq_error: torch.Tensor | None = None

    def _ensure_state(self, gauge_count: int) -> None:
        if self.count is None:
            zeros = torch.zeros(gauge_count, dtype=torch.float64)
            self.count = zeros.clone()
            self.sum_pred = zeros.clone()
            self.sum_obs = zeros.clone()
            self.sum_pred_sq = zeros.clone()
            self.sum_obs_sq = zeros.clone()
            self.sum_prod = zeros.clone()
            self.sum_sq_error = zeros.clone()
            return
        if int(self.count.numel()) != int(gauge_count):
            raise ValueError(
                f"Validation metrics expected {int(self.count.numel())} gauges but received {gauge_count}"
            )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(-1)
        if targets.ndim == 2:
            targets = targets.unsqueeze(-1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)

        if predictions.ndim != 3 or targets.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                "Validation metrics expect [B,T,G] predictions, targets, and mask. "
                f"Got prediction={tuple(predictions.shape)}, target={tuple(targets.shape)}, mask={tuple(mask.shape)}"
            )

        gauge_count = int(predictions.shape[-1])
        self._ensure_state(gauge_count)

        preds = predictions.detach().to(dtype=torch.float64)
        obs = targets.detach().to(dtype=torch.float64)
        valid = mask.detach().to(dtype=torch.bool) & torch.isfinite(preds) & torch.isfinite(obs)
        safe_preds = torch.where(valid, preds, torch.zeros_like(preds))
        safe_obs = torch.where(valid, obs, torch.zeros_like(obs))
        reduce_dims = (0, 1)

        assert self.count is not None
        assert self.sum_pred is not None
        assert self.sum_obs is not None
        assert self.sum_pred_sq is not None
        assert self.sum_obs_sq is not None
        assert self.sum_prod is not None
        assert self.sum_sq_error is not None

        self.count += valid.sum(dim=reduce_dims, dtype=torch.int64).to(dtype=torch.float64).cpu()
        self.sum_pred += safe_preds.sum(dim=reduce_dims).cpu()
        self.sum_obs += safe_obs.sum(dim=reduce_dims).cpu()
        self.sum_pred_sq += (safe_preds * safe_preds).sum(dim=reduce_dims).cpu()
        self.sum_obs_sq += (safe_obs * safe_obs).sum(dim=reduce_dims).cpu()
        self.sum_prod += (safe_preds * safe_obs).sum(dim=reduce_dims).cpu()
        self.sum_sq_error += ((safe_preds - safe_obs) ** 2).sum(dim=reduce_dims).cpu()

    def _metric_tensors(self) -> dict[str, torch.Tensor]:
        if self.count is None:
            return {}

        eps = torch.finfo(torch.float64).eps
        count = self.count.clamp(min=1.0)
        valid_points = self.count > 1.0
        mean_pred = self.sum_pred / count
        mean_obs = self.sum_obs / count
        var_pred = (self.sum_pred_sq / count - mean_pred.square()).clamp_min(0.0)
        var_obs = (self.sum_obs_sq / count - mean_obs.square()).clamp_min(0.0)
        std_pred = torch.sqrt(var_pred + eps)
        std_obs = torch.sqrt(var_obs + eps)
        covariance = self.sum_prod / count - mean_pred * mean_obs
        correlation = covariance / (std_pred * std_obs + eps)
        alpha = std_pred / (std_obs + eps)
        beta = (mean_pred + eps) / (mean_obs + eps)
        kge = 1.0 - torch.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

        sst = (self.sum_obs_sq - self.count * mean_obs.square()).clamp_min(0.0)
        nse = 1.0 - self.sum_sq_error / sst.clamp_min(eps)

        valid_kge = valid_points & (var_obs > eps) & (mean_obs.abs() > eps)
        valid_nse = valid_points & (sst > eps)
        mse = self.sum_sq_error / count

        return {
            "count": self.count,
            "mean_pred": mean_pred,
            "mean_obs": mean_obs,
            "correlation": correlation,
            "alpha": alpha,
            "beta": beta,
            "kge": kge,
            "nse": nse,
            "mse": mse,
            "valid_kge": valid_kge,
            "valid_nse": valid_nse,
        }

    def summarize(self) -> dict[str, float | int | list[dict[str, float | int | str]]]:
        metrics = self._metric_tensors()
        if not metrics:
            return {
                "mean_kge": float("nan"),
                "mean_nse": float("nan"),
                "valid_kge_gauges": 0,
                "valid_nse_gauges": 0,
                "per_gauge_metrics": [],
            }
        valid_kge = metrics["valid_kge"].to(dtype=torch.bool)
        valid_nse = metrics["valid_nse"].to(dtype=torch.bool)
        mean_correlation = float(metrics["correlation"][valid_kge].mean().item()) if bool(valid_kge.any()) else float("nan")
        mean_alpha = float(metrics["alpha"][valid_kge].mean().item()) if bool(valid_kge.any()) else float("nan")
        mean_beta = float(metrics["beta"][valid_kge].mean().item()) if bool(valid_kge.any()) else float("nan")
        component_mean_kge = (
            1.0
            - float(
                torch.sqrt(
                    (torch.as_tensor(mean_correlation, dtype=torch.float64) - 1.0) ** 2
                    + (torch.as_tensor(mean_alpha, dtype=torch.float64) - 1.0) ** 2
                    + (torch.as_tensor(mean_beta, dtype=torch.float64) - 1.0) ** 2
                ).item()
            )
            if bool(valid_kge.any())
            else float("nan")
        )
        return {
            "mean_kge": float(metrics["kge"][valid_kge].mean().item()) if bool(valid_kge.any()) else float("nan"),
            "mean_nse": float(metrics["nse"][valid_nse].mean().item()) if bool(valid_nse.any()) else float("nan"),
            "component_mean_kge": component_mean_kge,
            "mean_correlation": mean_correlation,
            "mean_alpha": mean_alpha,
            "mean_beta": mean_beta,
            "valid_kge_gauges": int(valid_kge.sum().item()),
            "valid_nse_gauges": int(valid_nse.sum().item()),
            "per_gauge_metrics": self.per_gauge_rows(metrics),
        }

    def per_gauge_rows(
        self,
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> list[dict[str, float | int | str]]:
        metrics = metrics or self._metric_tensors()
        if not metrics:
            return []

        rows: list[dict[str, float | int | str]] = []
        count = metrics["count"]
        valid_kge = metrics["valid_kge"].to(dtype=torch.bool)
        valid_nse = metrics["valid_nse"].to(dtype=torch.bool)
        for gauge_index in range(int(count.numel())):
            gauge_id = self.gauge_ids[gauge_index] if gauge_index < len(self.gauge_ids) else str(gauge_index)
            has_kge = bool(valid_kge[gauge_index].item())
            has_nse = bool(valid_nse[gauge_index].item())
            rows.append(
                {
                    "gauge_index": gauge_index,
                    "gauge_id": gauge_id,
                    "valid_points": int(count[gauge_index].item()),
                    "kge": float(metrics["kge"][gauge_index].item()) if has_kge else float("nan"),
                    "nse": float(metrics["nse"][gauge_index].item()) if has_nse else float("nan"),
                    "correlation": float(metrics["correlation"][gauge_index].item()) if has_kge else float("nan"),
                    "alpha": float(metrics["alpha"][gauge_index].item()) if has_kge else float("nan"),
                    "beta": float(metrics["beta"][gauge_index].item()) if has_kge else float("nan"),
                    "mean_prediction": float(metrics["mean_pred"][gauge_index].item()) if has_kge else float("nan"),
                    "mean_observation": float(metrics["mean_obs"][gauge_index].item()) if has_kge else float("nan"),
                    "mse": float(metrics["mse"][gauge_index].item()) if int(count[gauge_index].item()) > 0 else float("nan"),
                }
            )
        return rows


class RunoffRoutingPipeline(nn.Module):
    """LSTM runoff generator followed by optional grid-to-graph transfer and GNN routing."""

    def __init__(
        self,
        *,
        runoff_model: nn.Module,
        routing_model: nn.Module,
        runoff_transfer: nn.Module | None = None,
        runoff_device: torch.device,
        routing_device: torch.device,
        prediction_key: str = "runoff_total",
    ):
        super().__init__()
        self.runoff_model = runoff_model
        self.runoff_transfer = runoff_transfer
        self.routing_model = routing_model
        self.runoff_device = runoff_device
        self.routing_device = routing_device
        self.prediction_key = prediction_key

    @staticmethod
    def _context_steps(batch: Mapping[str, Any], key: str) -> int:
        value = batch.get(key, 0)
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].detach().cpu().item())
        return int(value or 0)

    def _runoff_model_batch_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(key) for key in getattr(self.runoff_model, "input_keys", ())))

    def _runoff_transfer_batch_keys(self) -> tuple[str, ...]:
        if self.runoff_transfer is None:
            return ()
        keys = list(getattr(self.runoff_transfer, "input_keys", ()))
        graph_key = getattr(self.runoff_transfer, "graph_key", None)
        if graph_key:
            keys.append(str(graph_key))
        return tuple(dict.fromkeys(str(key) for key in keys))

    def _routing_batch_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for attr_name in ("dynamic_input_keys", "static_input_keys", "input_keys"):
            keys.extend(str(key) for key in getattr(self.routing_model, attr_name, ()))
        for attr_name in ("graph_key", "weight_key"):
            value = getattr(self.routing_model, attr_name, None)
            if value:
                keys.append(str(value))
        return tuple(dict.fromkeys(keys))

    def _batch_view_for_runoff_model(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        keys = self._runoff_model_batch_keys()
        subset = batch if not keys else _subset_mapping(batch, keys)
        return _move_mapping_to_device(subset, self.runoff_device)

    def _batch_view_for_runoff_transfer(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.runoff_transfer is None:
            return {}
        keys = self._runoff_transfer_batch_keys()
        subset = batch if not keys else _subset_mapping(batch, keys)
        return _move_mapping_to_device(subset, self.routing_device)

    def _batch_view_for_routing(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        keys = self._routing_batch_keys()
        subset = batch if not keys else _subset_mapping(batch, keys)
        return _move_mapping_to_device(subset, self.routing_device)

    def _route_outputs(self, runoff_outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> torch.Tensor:
        routed_outputs = _move_mapping_values_to_device(runoff_outputs, self.routing_device)
        if self.runoff_transfer is not None:
            routed_outputs = self.runoff_transfer(
                routed_outputs,
                self._batch_view_for_runoff_transfer(batch),
            )
        routed = self.routing_model(routed_outputs, self._batch_view_for_routing(batch))
        if isinstance(routed, Mapping):
            if "predictions" in routed:
                return routed["predictions"]
            if "y_hat" in routed:
                return routed["y_hat"]
            raise KeyError("routing_model mapping output must contain 'predictions' or 'y_hat'")
        return routed

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        runoff_outputs = self.runoff_model(self._batch_view_for_runoff_model(batch))
        if not isinstance(runoff_outputs, Mapping):
            raise TypeError("runoff_model must return a mapping of runoff output tensors")
        routing_runoff_outputs = _trim_runoff_outputs_for_routing(runoff_outputs, batch)
        prediction = self._route_outputs(routing_runoff_outputs, batch)
        prediction = _trim_prediction_context(prediction, batch)
        return {
            self.prediction_key: prediction,
            "runoff_outputs": runoff_outputs,
        }


class LSTMGNNTrainer:
    """Small standalone trainer for runoff-generation and graph-routing experiments."""

    def __init__(self, config: RoutingConfig):
        self.config = config
        training_cfg = config.section("training")
        _seed_everything(int(training_cfg.get("seed", 42)))

        self.run_dir = Path(config.run_dir or Path("runs") / config.experiment_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config.dump_config(self.run_dir)

        logger.info("Initializing train dataset")
        self.train_dataset = RoutingDataset(config, "train")
        logger.info("Initializing validation dataset")
        self.val_dataset = RoutingDataset(config, "validation", scaler=self.train_dataset.scaler)
        batcher = RoutingBatcher(self.train_dataset, self.val_dataset)
        self.train_loader = batcher.train_loader()
        self.val_loader = batcher.validation_loader()
        logger.info(
            "Data loaders ready | train_samples=%s | val_samples=%s | train_batches=%s | val_batches=%s | batch_size=%s",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.train_loader),
            len(self.val_loader),
            config.batch_size,
        )

        example_batch = next(iter(self.train_loader))
        execution_cfg = config.section("execution")
        default_device = torch.device(config.device if torch.cuda.is_available() or str(config.device) == "cpu" else "cpu")
        self.runoff_device = torch.device(execution_cfg.get("runoff_model_device", default_device))
        self.routing_device = torch.device(execution_cfg.get("routing_device", default_device))

        runoff_model = build_runoff_model(config, example_batch=example_batch, device=self.runoff_device)
        runoff_transfer = build_runoff_transfer_model(config, example_batch=example_batch, device=self.routing_device)
        routing_model = build_routing_model(config, example_batch=example_batch, device=self.routing_device)
        self.model = RunoffRoutingPipeline(
            runoff_model=runoff_model,
            runoff_transfer=runoff_transfer,
            routing_model=routing_model,
            runoff_device=self.runoff_device,
            routing_device=self.routing_device,
            prediction_key=str(training_cfg.get("default_prediction_key", "runoff_total")),
        )

        self.loss_fn = get_loss_function(config)
        self.use_amp = bool(training_cfg.get("use_amp", False)) and self.routing_device.type == "cuda"
        self.amp_device_type = "cuda" if self.routing_device.type == "cuda" else self.routing_device.type
        self.grad_scaler = GradScaler(self.amp_device_type, enabled=self.use_amp)
        self.grad_clip_norm = training_cfg.get("grad_clip_norm")
        self.skip_nonfinite_batches = bool(training_cfg.get("skip_nonfinite_batches", True))
        self.show_progress = bool(training_cfg.get("show_progress", True))
        self.leave_progress = bool(training_cfg.get("leave_progress", False))
        self.prediction_key = str(training_cfg.get("default_prediction_key", "runoff_total"))

        self.optimizer = self._build_optimizer(float(training_cfg.get("learning_rate", 1e-3)))
        self.history_path = self.run_dir / "training_history.csv"
        self.validation_metrics_path = self.run_dir / "validation_metrics.csv"
        self.per_gauge_validation_metrics_path = self.run_dir / "validation_metrics_by_gauge.csv"
        checkpoint_cfg = config.section("checkpoint")
        self.save_checkpoints = bool(checkpoint_cfg.get("enabled", True))
        self.save_best_checkpoint = bool(checkpoint_cfg.get("save_best", True))
        self.save_last_checkpoint = bool(checkpoint_cfg.get("save_last", True))
        self.save_stage_best_checkpoint = bool(checkpoint_cfg.get("save_stage_best", False))
        self.save_final_stage_best_checkpoint = bool(checkpoint_cfg.get("save_final_stage_best", False))
        self.restore_stage_best = bool(checkpoint_cfg.get("restore_stage_best", False))
        self.restore_stage_best_optimizer = bool(checkpoint_cfg.get("restore_stage_best_optimizer", True))
        self.best_checkpoint_path = self.run_dir / str(checkpoint_cfg.get("best_file", "best_model.pt"))
        self.last_checkpoint_path = self.run_dir / str(checkpoint_cfg.get("last_file", "last_model.pt"))
        self.stage_best_file_template = str(checkpoint_cfg.get("stage_best_file_template", "best_stage_{stage_index}_{stage_name}.pt"))
        self.final_stage_best_checkpoint_path = self.run_dir / str(checkpoint_cfg.get("final_stage_best_file", "best_final_stage_model.pt"))
        self.best_val_loss = float("inf")
        self.metric_target_transform = "identity"
        self.metric_target_means: torch.Tensor | None = None
        self.metric_target_stds: torch.Tensor | None = None
        self._configure_validation_metric_scaler()
        if hasattr(self.loss_fn, "configure_target_scaler"):
            self.loss_fn.configure_target_scaler(
                getattr(self.train_dataset, "target_scaler", None),
                device=self.routing_device,
            )
        self._log_example_batch_probe(example_batch)

    def _build_optimizer(self, learning_rate: float):
        training_cfg = self.config.section("training")
        optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        params = [param for param in self.model.parameters() if param.requires_grad]
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        raise ValueError("training.optimizer must be one of: adam, adamw")

    @staticmethod
    def _tensor_shape(value: Any) -> tuple[int, ...] | None:
        if torch.is_tensor(value):
            return tuple(int(v) for v in value.shape)
        return None

    def _log_example_batch_probe(self, batch: Mapping[str, Any]) -> None:
        graph = batch.get("routing_graph")
        graph_nodes = graph.get("num_nodes") if isinstance(graph, Mapping) else None
        graph_edges = graph.get("num_edges") if isinstance(graph, Mapping) else None
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                outputs = self.model(batch)
        finally:
            self.model.train(was_training)

        runoff_shapes = {
            key: tuple(int(v) for v in value.shape)
            for key, value in (outputs.get("runoff_outputs", {}) or {}).items()
            if torch.is_tensor(value)
        }
        logger.info(
            "Example batch probe | x_forcing_ml=%s | x_static_ml=%s | y=%s | loss_mask=%s | routing_graph(nodes=%s, edges=%s) | runoff_outputs=%s | prediction=%s",
            self._tensor_shape(batch.get("x_forcing_ml")),
            self._tensor_shape(batch.get("x_static_ml")),
            self._tensor_shape(batch.get("y")),
            self._tensor_shape(batch.get("loss_mask")),
            graph_nodes,
            graph_edges,
            runoff_shapes,
            self._tensor_shape(outputs.get(self.prediction_key)),
        )

    def _set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _stage_best_checkpoint_path(self, stage_index: int, stage_name: str) -> Path:
        filename = self.stage_best_file_template.format(
            stage_index=int(stage_index),
            stage_name=_safe_checkpoint_token(stage_name),
        )
        return self.run_dir / filename

    def _configure_validation_metric_scaler(self) -> None:
        scaler = getattr(self.val_dataset, "target_scaler", None)
        if not scaler:
            return
        means = np.asarray(scaler.get("means"), dtype=np.float32)
        stds = np.asarray(scaler.get("stds"), dtype=np.float32)
        if means.ndim == 2 and means.shape[1] == 1:
            means = means[:, 0]
            stds = stds[:, 0]
        elif means.ndim != 1:
            logger.warning(
                "Validation KGE/NSE inverse transform supports a single target variable per gauge; "
                "received scaler shape %s. Metrics will stay in normalized units.",
                tuple(int(v) for v in means.shape),
            )
            return
        self.metric_target_transform = str(scaler.get("transform", "identity")).lower()
        self.metric_target_means = torch.as_tensor(means, dtype=torch.float32, device=self.routing_device)
        self.metric_target_stds = torch.as_tensor(stds, dtype=torch.float32, device=self.routing_device)

    def _inverse_transform_metric_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.metric_target_means is None or self.metric_target_stds is None:
            return tensor
        if tensor.ndim != 3:
            raise ValueError(
                "Validation target inverse transform expects [B,T,G] tensors, "
                f"got shape {tuple(tensor.shape)}"
            )
        if int(tensor.shape[-1]) != int(self.metric_target_means.numel()):
            raise ValueError(
                f"Validation target scaler has {int(self.metric_target_means.numel())} gauges, "
                f"but tensor shape is {tuple(tensor.shape)}"
            )
        means = self.metric_target_means.to(device=tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        stds = self.metric_target_stds.to(device=tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        values = tensor * stds + means
        if self.metric_target_transform in {"", "none", "identity"}:
            return values
        if self.metric_target_transform == "log1p":
            return torch.expm1(values)
        raise ValueError(
            f"Unsupported validation target transform '{self.metric_target_transform}' for KGE/NSE reporting."
        )

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        stage_index: int,
        stage_epoch: int,
        stage_name: str,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        is_best: bool,
        val_kge: float | None = None,
        val_nse: float | None = None,
        valid_kge_gauges: int | None = None,
        valid_nse_gauges: int | None = None,
        best_val_loss: float | None = None,
    ) -> dict[str, Any]:
        recorded_best = best_val_loss
        if recorded_best is None:
            recorded_best = min(self.best_val_loss, val_loss) if is_best else self.best_val_loss
        return {
            "epoch": int(epoch),
            "stage_index": int(stage_index),
            "stage_epoch": int(stage_epoch),
            "stage_name": str(stage_name),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_kge": None if val_kge is None else float(val_kge),
            "val_nse": None if val_nse is None else float(val_nse),
            "valid_kge_gauges": None if valid_kge_gauges is None else int(valid_kge_gauges),
            "valid_nse_gauges": None if valid_nse_gauges is None else int(valid_nse_gauges),
            "learning_rate": float(learning_rate),
            "best_val_loss": float(recorded_best),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_path": str(self.run_dir / "config.yml"),
        }

    def _save_checkpoint(
        self,
        file_path: Path,
        *,
        epoch: int,
        stage_index: int,
        stage_epoch: int,
        stage_name: str,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        is_best: bool,
        val_kge: float | None = None,
        val_nse: float | None = None,
        valid_kge_gauges: int | None = None,
        valid_nse_gauges: int | None = None,
        best_val_loss: float | None = None,
    ) -> None:
        payload = self._checkpoint_payload(
            epoch=epoch,
            stage_index=stage_index,
            stage_epoch=stage_epoch,
            stage_name=stage_name,
            train_loss=train_loss,
            val_loss=val_loss,
            val_kge=val_kge,
            val_nse=val_nse,
            valid_kge_gauges=valid_kge_gauges,
            valid_nse_gauges=valid_nse_gauges,
            learning_rate=learning_rate,
            is_best=is_best,
            best_val_loss=best_val_loss,
        )
        torch.save(payload, file_path)
        logger.info(
            "Saved %scheckpoint to %s (epoch=%s, stage=%s, val_loss=%.6f)",
            "best " if is_best else "",
            file_path,
            epoch,
            stage_name,
            val_loss,
        )

    def _load_checkpoint(self, file_path: Path, *, restore_optimizer: bool) -> None:
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if restore_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(
            "Restored checkpoint from %s (epoch=%s, stage=%s, val_loss=%.6f)",
            file_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("stage_name", "?"),
            float(checkpoint.get("val_loss", float("nan"))),
        )

    @staticmethod
    def _positive_weight_mapping(raw_mapping: Any) -> dict[str, float]:
        if raw_mapping in (None, "", {}):
            return {}
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("Gauge weights must be provided as a YAML mapping of gauge_id: positive_weight.")
        result: dict[str, float] = {}
        for gauge_id, raw_weight in raw_mapping.items():
            weight = float(raw_weight)
            if weight <= 0.0:
                raise ValueError(f"Gauge weight for '{gauge_id}' must be positive, got {raw_weight!r}.")
            result[str(gauge_id)] = weight
        return result

    def _stage_gauge_weight_config(
        self,
        stage: Mapping[str, Any] | None,
    ) -> tuple[dict[str, float], float | None, set[str]]:
        curriculum_cfg = self.config.section("curriculum")
        gauge_weights = self._positive_weight_mapping(curriculum_cfg.get("gauge_weights"))
        if stage:
            gauge_weights.update(self._positive_weight_mapping(stage.get("gauge_weights")))

        outlet_weight_raw = curriculum_cfg.get("outlet_gauge_weight")
        if stage and stage.get("outlet_gauge_weight") is not None:
            outlet_weight_raw = stage.get("outlet_gauge_weight")
        outlet_weight = None if outlet_weight_raw in (None, "") else float(outlet_weight_raw)
        if outlet_weight is not None and outlet_weight <= 0.0:
            raise ValueError(f"Outlet gauge weight must be positive, got {outlet_weight_raw!r}.")

        outlet_gauges = curriculum_cfg.get("independent_outlet_gauges", [])
        if stage and stage.get("outlet_gauges") is not None:
            outlet_gauges = stage.get("outlet_gauges")
        outlet_gauge_set = {str(value) for value in (outlet_gauges or [])}
        return gauge_weights, outlet_weight, outlet_gauge_set

    def _gauge_mask_and_weights(self, batch: Mapping[str, Any], stage: Mapping[str, Any] | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not stage:
            return None, None
        gauges = [str(value) for value in stage.get("gauges", [])]
        if not gauges:
            return None, None
        catchment_ids = [str(value) for value in batch["x_info"][0].get("catchment_ids", [])]
        device = self.routing_device
        mask = torch.zeros(len(catchment_ids), dtype=torch.bool, device=device)
        weights = torch.ones(len(catchment_ids), dtype=torch.float32, device=device)
        stage_gauge_weights, outlet_gauge_weight, outlet_gauges = self._stage_gauge_weight_config(stage)
        for gauge in gauges:
            if gauge in catchment_ids:
                index = catchment_ids.index(gauge)
                mask[index] = True
                weight = stage_gauge_weights.get(gauge, 1.0)
                if outlet_gauge_weight is not None and gauge in outlet_gauges:
                    weight *= outlet_gauge_weight
                weights[index] = float(weight)
        if not bool(mask.any()):
            raise ValueError(f"Curriculum stage '{stage.get('name', '')}' has no gauges present in this batch.")
        return mask, weights

    def _prepare_loss_tensors(
        self,
        batch: Mapping[str, Any],
        prediction: torch.Tensor,
        *,
        stage: Mapping[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target = batch["y"].to(device=self.routing_device, non_blocking=True)
        target_mask = batch["target_mask"].to(device=self.routing_device, non_blocking=True)
        if target.ndim == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if target_mask.ndim == 4 and target_mask.shape[-1] == 1:
            target_mask = target_mask.squeeze(-1)

        if prediction.shape[1] != target.shape[1]:
            common = min(int(prediction.shape[1]), int(target.shape[1]))
            prediction = prediction[:, -common:]
            target = target[:, -common:]
            target_mask = target_mask[:, -common:]

        loss_time_mask = batch["loss_mask"].to(device=self.routing_device, non_blocking=True)
        if loss_time_mask.ndim == 2:
            loss_time_mask = loss_time_mask[:, -target.shape[1]:].unsqueeze(-1).expand_as(target_mask)
        valid_mask = target_mask & loss_time_mask.to(dtype=torch.bool)

        gauge_mask, gauge_weights = self._gauge_mask_and_weights(batch, stage)
        if gauge_mask is not None:
            valid_mask = valid_mask & gauge_mask.view(1, 1, -1)
        return prediction, target, valid_mask, gauge_weights

    def _run_epoch(
        self,
        loader,
        *,
        train: bool,
        stage: Mapping[str, Any] | None = None,
        epoch_label: str = "",
        compute_metrics: bool = False,
    ) -> dict[str, Any]:
        self.model.train(mode=train)
        running = 0.0
        count = 0
        metric_accumulator: _PerGaugeValidationMetrics | None = None
        iterator = tqdm(
            loader,
            desc=epoch_label,
            dynamic_ncols=True,
            leave=self.leave_progress,
            disable=not self.show_progress,
        )
        for batch in iterator:
            with torch.set_grad_enabled(train):
                with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                    outputs = self.model(batch)
                    prediction, target, mask, weights = self._prepare_loss_tensors(
                        batch,
                        outputs[self.prediction_key],
                        stage=stage,
                    )
                    loss = self.loss_fn(prediction, target, mask=mask, weights=weights)

            if not torch.isfinite(loss):
                if self.skip_nonfinite_batches:
                    logger.warning("Skipping %s batch with non-finite loss: %s", "train" if train else "eval", float(loss.detach().cpu()))
                    continue
                raise RuntimeError(f"Non-finite loss: {float(loss.detach().cpu())}")

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if self.grad_clip_norm not in (None, 0):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
                    if not torch.isfinite(grad_norm):
                        if self.skip_nonfinite_batches:
                            logger.warning("Skipping train batch with non-finite gradient norm %s", float(grad_norm.detach().cpu()))
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        raise RuntimeError(f"Non-finite gradient norm: {float(grad_norm.detach().cpu())}")
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

            value = float(loss.detach().cpu())
            running += value
            count += 1
            if compute_metrics:
                if metric_accumulator is None:
                    x_info = batch.get("x_info")
                    info0 = x_info[0] if isinstance(x_info, list) and x_info else x_info
                    catchment_ids = []
                    if isinstance(info0, Mapping):
                        catchment_ids = [str(value) for value in info0.get("catchment_ids", [])]
                    metric_accumulator = _PerGaugeValidationMetrics(catchment_ids)
                metric_prediction = self._inverse_transform_metric_tensor(prediction.detach())
                metric_target = self._inverse_transform_metric_tensor(target.detach())
                metric_accumulator.update(metric_prediction, metric_target, mask)
            iterator.set_postfix(avg_loss=f"{running / max(count, 1):.4f}", batch_loss=f"{value:.4f}")
        if count == 0:
            raise RuntimeError("No finite batches were available for this epoch.")
        result: dict[str, Any] = {"loss": running / count}
        if compute_metrics and metric_accumulator is not None:
            result.update(metric_accumulator.summarize())
        return result

    def _curriculum_stages(self) -> list[dict[str, Any]]:
        curriculum_cfg = self.config.section("curriculum")
        if not bool(curriculum_cfg.get("enabled", False)):
            return [{"name": "all_gauges", "epochs": int(self.config.section("training").get("epochs", 1)), "gauges": []}]
        stages = [dict(stage) for stage in curriculum_cfg.get("stages", [])]
        if not stages:
            raise ValueError("curriculum.enabled=true requires curriculum.stages")
        return stages

    def train_and_validate(self) -> None:
        training_cfg = self.config.section("training")
        base_lr = float(training_cfg.get("learning_rate", 1e-3))
        lr_decay = float(self.config.section("curriculum").get("stage_learning_rate_decay", 1.0))
        patience = training_cfg.get("early_stopping_patience")
        early_stopper = None if patience in (None, 0) else EarlyStopper(int(patience), float(training_cfg.get("early_stopping_min_delta", 0.0)))

        history_fields = [
            "epoch",
            "stage",
            "stage_index",
            "stage_epoch",
            "train_loss",
            "val_loss",
            "val_kge",
            "val_kge_component_mean",
            "val_kge_mean_correlation",
            "val_kge_mean_alpha",
            "val_kge_mean_beta",
            "val_nse",
            "valid_kge_gauges",
            "valid_nse_gauges",
            "learning_rate",
        ]
        metric_fields = [
            "epoch",
            "stage",
            "stage_index",
            "stage_epoch",
            "val_loss",
            "val_kge",
            "val_kge_component_mean",
            "val_kge_mean_correlation",
            "val_kge_mean_alpha",
            "val_kge_mean_beta",
            "val_nse",
            "valid_kge_gauges",
            "valid_nse_gauges",
            "learning_rate",
        ]
        per_gauge_metric_fields = [
            "epoch",
            "stage",
            "stage_index",
            "stage_epoch",
            "gauge_index",
            "gauge_id",
            "valid_points",
            "kge",
            "nse",
            "correlation",
            "alpha",
            "beta",
            "mean_prediction",
            "mean_observation",
            "mse",
            "learning_rate",
        ]
        stages = self._curriculum_stages()
        write_header = _ensure_csv_schema(self.history_path, history_fields)
        metrics_write_header = _ensure_csv_schema(self.validation_metrics_path, metric_fields)
        per_gauge_metrics_write_header = _ensure_csv_schema(self.per_gauge_validation_metrics_path, per_gauge_metric_fields)
        with (
            self.history_path.open("a", newline="") as history_fp,
            self.validation_metrics_path.open("a", newline="") as metrics_fp,
            self.per_gauge_validation_metrics_path.open("a", newline="") as per_gauge_metrics_fp,
        ):
            history_writer = csv.DictWriter(
                history_fp,
                fieldnames=history_fields,
            )
            if write_header:
                history_writer.writeheader()
            metrics_writer = csv.DictWriter(
                metrics_fp,
                fieldnames=metric_fields,
            )
            if metrics_write_header:
                metrics_writer.writeheader()
            per_gauge_metrics_writer = csv.DictWriter(
                per_gauge_metrics_fp,
                fieldnames=per_gauge_metric_fields,
            )
            if per_gauge_metrics_write_header:
                per_gauge_metrics_writer.writeheader()

            global_epoch = 0
            stage_count = len(stages)
            for stage_index, stage in enumerate(stages, start=1):
                stage_name = str(stage.get("name", f"stage_{stage_index}"))
                stage_epochs = int(stage.get("epochs", training_cfg.get("epochs", 1)))
                stage_lr = float(stage.get("learning_rate", base_lr * (lr_decay ** (stage_index - 1))))
                stage_best_val_loss = float("inf")
                stage_best_checkpoint_path = self._stage_best_checkpoint_path(stage_index, stage_name)
                self._set_learning_rate(stage_lr)
                if early_stopper is not None and self.config.section("curriculum").get("early_stopping_scope", "stage") == "stage":
                    early_stopper.reset()

                for stage_epoch in range(1, stage_epochs + 1):
                    global_epoch += 1
                    train_metrics = self._run_epoch(
                        self.train_loader,
                        train=True,
                        stage=stage,
                        epoch_label=f"train {global_epoch}",
                    )
                    train_loss = float(train_metrics["loss"])
                    with torch.no_grad():
                        val_metrics = self._run_epoch(
                            self.val_loader,
                            train=False,
                            stage=stage,
                            epoch_label=f"validation {global_epoch}",
                            compute_metrics=True,
                        )
                    val_loss = float(val_metrics["loss"])
                    val_kge = float(val_metrics.get("mean_kge", float("nan")))
                    val_kge_component_mean = float(val_metrics.get("component_mean_kge", float("nan")))
                    val_kge_mean_correlation = float(val_metrics.get("mean_correlation", float("nan")))
                    val_kge_mean_alpha = float(val_metrics.get("mean_alpha", float("nan")))
                    val_kge_mean_beta = float(val_metrics.get("mean_beta", float("nan")))
                    val_nse = float(val_metrics.get("mean_nse", float("nan")))
                    valid_kge_gauges = int(val_metrics.get("valid_kge_gauges", 0))
                    valid_nse_gauges = int(val_metrics.get("valid_nse_gauges", 0))
                    per_gauge_metrics = val_metrics.get("per_gauge_metrics", [])
                    logger.info(
                        "Epoch %s | stage=%s | train_loss=%.6f | val_loss=%.6f | val_kge=%.6f | val_nse=%.6f | lr=%.6g",
                        global_epoch,
                        stage_name,
                        train_loss,
                        val_loss,
                        val_kge,
                        val_nse,
                        stage_lr,
                    )
                    history_writer.writerow(
                        {
                            "epoch": global_epoch,
                            "stage": stage_name,
                            "stage_index": stage_index,
                            "stage_epoch": stage_epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_kge": val_kge,
                            "val_kge_component_mean": val_kge_component_mean,
                            "val_kge_mean_correlation": val_kge_mean_correlation,
                            "val_kge_mean_alpha": val_kge_mean_alpha,
                            "val_kge_mean_beta": val_kge_mean_beta,
                            "val_nse": val_nse,
                            "valid_kge_gauges": valid_kge_gauges,
                            "valid_nse_gauges": valid_nse_gauges,
                            "learning_rate": stage_lr,
                        }
                    )
                    history_fp.flush()
                    metrics_writer.writerow(
                        {
                            "epoch": global_epoch,
                            "stage": stage_name,
                            "stage_index": stage_index,
                            "stage_epoch": stage_epoch,
                            "val_loss": val_loss,
                            "val_kge": val_kge,
                            "val_kge_component_mean": val_kge_component_mean,
                            "val_kge_mean_correlation": val_kge_mean_correlation,
                            "val_kge_mean_alpha": val_kge_mean_alpha,
                            "val_kge_mean_beta": val_kge_mean_beta,
                            "val_nse": val_nse,
                            "valid_kge_gauges": valid_kge_gauges,
                            "valid_nse_gauges": valid_nse_gauges,
                            "learning_rate": stage_lr,
                        }
                    )
                    metrics_fp.flush()
                    for row in per_gauge_metrics:
                        per_gauge_metrics_writer.writerow(
                            {
                                "epoch": global_epoch,
                                "stage": stage_name,
                                "stage_index": stage_index,
                                "stage_epoch": stage_epoch,
                                "gauge_index": row.get("gauge_index", ""),
                                "gauge_id": row.get("gauge_id", ""),
                                "valid_points": row.get("valid_points", ""),
                                "kge": row.get("kge", ""),
                                "nse": row.get("nse", ""),
                                "correlation": row.get("correlation", ""),
                                "alpha": row.get("alpha", ""),
                                "beta": row.get("beta", ""),
                                "mean_prediction": row.get("mean_prediction", ""),
                                "mean_observation": row.get("mean_observation", ""),
                                "mse": row.get("mse", ""),
                                "learning_rate": stage_lr,
                            }
                        )
                    per_gauge_metrics_fp.flush()

                    if self.save_checkpoints:
                        if self.save_last_checkpoint:
                            self._save_checkpoint(
                                self.last_checkpoint_path,
                                epoch=global_epoch,
                                stage_index=stage_index,
                                stage_epoch=stage_epoch,
                                stage_name=stage_name,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                learning_rate=stage_lr,
                                is_best=False,
                                val_kge=val_kge,
                                val_nse=val_nse,
                                valid_kge_gauges=valid_kge_gauges,
                                valid_nse_gauges=valid_nse_gauges,
                            )
                        if self.save_best_checkpoint and val_loss < self.best_val_loss:
                            self.best_val_loss = float(val_loss)
                            self._save_checkpoint(
                                self.best_checkpoint_path,
                                epoch=global_epoch,
                                stage_index=stage_index,
                                stage_epoch=stage_epoch,
                                stage_name=stage_name,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                learning_rate=stage_lr,
                                is_best=True,
                                val_kge=val_kge,
                                val_nse=val_nse,
                                valid_kge_gauges=valid_kge_gauges,
                                valid_nse_gauges=valid_nse_gauges,
                                best_val_loss=self.best_val_loss,
                            )
                        if val_loss < stage_best_val_loss:
                            stage_best_val_loss = float(val_loss)
                            if self.save_stage_best_checkpoint:
                                self._save_checkpoint(
                                    stage_best_checkpoint_path,
                                    epoch=global_epoch,
                                    stage_index=stage_index,
                                    stage_epoch=stage_epoch,
                                    stage_name=stage_name,
                                    train_loss=train_loss,
                                    val_loss=val_loss,
                                    learning_rate=stage_lr,
                                    is_best=True,
                                    val_kge=val_kge,
                                    val_nse=val_nse,
                                    valid_kge_gauges=valid_kge_gauges,
                                    valid_nse_gauges=valid_nse_gauges,
                                    best_val_loss=stage_best_val_loss,
                                )
                            if self.save_final_stage_best_checkpoint and stage_index == stage_count:
                                self._save_checkpoint(
                                    self.final_stage_best_checkpoint_path,
                                    epoch=global_epoch,
                                    stage_index=stage_index,
                                    stage_epoch=stage_epoch,
                                    stage_name=stage_name,
                                    train_loss=train_loss,
                                    val_loss=val_loss,
                                    learning_rate=stage_lr,
                                    is_best=True,
                                    val_kge=val_kge,
                                    val_nse=val_nse,
                                    valid_kge_gauges=valid_kge_gauges,
                                    valid_nse_gauges=valid_nse_gauges,
                                    best_val_loss=stage_best_val_loss,
                                )

                    if early_stopper is not None and early_stopper.step(val_loss):
                        logger.info("Early stopping triggered for stage '%s'.", stage_name)
                        break

                if self.restore_stage_best:
                    if self.save_checkpoints and self.save_stage_best_checkpoint and stage_best_checkpoint_path.exists():
                        self._load_checkpoint(
                            stage_best_checkpoint_path,
                            restore_optimizer=self.restore_stage_best_optimizer,
                        )
                        self._set_learning_rate(stage_lr)
                    else:
                        logger.warning(
                            "restore_stage_best=true but no stage-best checkpoint was available for stage '%s'.",
                            stage_name,
                        )
