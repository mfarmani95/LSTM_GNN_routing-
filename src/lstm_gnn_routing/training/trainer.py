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
        self._log_example_batch_probe(example_batch)

    def _build_optimizer(self, learning_rate: float):
        training_cfg = self.config.section("training")
        optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
        params = [param for param in self.model.parameters() if param.requires_grad]
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=learning_rate)
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=learning_rate)
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

    def _gauge_mask_and_weights(self, batch: Mapping[str, Any], stage: Mapping[str, Any] | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not stage:
            return None, None
        gauges = [str(value) for value in stage.get("gauges", [])]
        if not gauges:
            return None, None
        catchment_ids = [str(value) for value in batch["x_info"][0].get("catchment_ids", [])]
        device = self.routing_device
        mask = torch.zeros(len(catchment_ids), dtype=torch.bool, device=device)
        for gauge in gauges:
            if gauge in catchment_ids:
                mask[catchment_ids.index(gauge)] = True
        if not bool(mask.any()):
            raise ValueError(f"Curriculum stage '{stage.get('name', '')}' has no gauges present in this batch.")
        return mask, mask.to(dtype=torch.float32)

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

    def _run_epoch(self, loader, *, train: bool, stage: Mapping[str, Any] | None = None, epoch_label: str = "") -> float:
        self.model.train(mode=train)
        running = 0.0
        count = 0
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
            iterator.set_postfix(avg_loss=f"{running / max(count, 1):.4f}", batch_loss=f"{value:.4f}")
        if count == 0:
            raise RuntimeError("No finite batches were available for this epoch.")
        return running / count

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

        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["epoch", "stage", "stage_index", "stage_epoch", "train_loss", "val_loss", "learning_rate"],
            )
            if write_header:
                writer.writeheader()

            global_epoch = 0
            for stage_index, stage in enumerate(self._curriculum_stages(), start=1):
                stage_name = str(stage.get("name", f"stage_{stage_index}"))
                stage_epochs = int(stage.get("epochs", training_cfg.get("epochs", 1)))
                stage_lr = float(stage.get("learning_rate", base_lr * (lr_decay ** (stage_index - 1))))
                self._set_learning_rate(stage_lr)
                if early_stopper is not None and self.config.section("curriculum").get("early_stopping_scope", "stage") == "stage":
                    early_stopper.reset()

                for stage_epoch in range(1, stage_epochs + 1):
                    global_epoch += 1
                    train_loss = self._run_epoch(
                        self.train_loader,
                        train=True,
                        stage=stage,
                        epoch_label=f"train {global_epoch}",
                    )
                    with torch.no_grad():
                        val_loss = self._run_epoch(
                            self.val_loader,
                            train=False,
                            stage=stage,
                            epoch_label=f"validation {global_epoch}",
                        )
                    logger.info(
                        "Epoch %s | stage=%s | train_loss=%.6f | val_loss=%.6f | lr=%.6g",
                        global_epoch,
                        stage_name,
                        train_loss,
                        val_loss,
                        stage_lr,
                    )
                    writer.writerow(
                        {
                            "epoch": global_epoch,
                            "stage": stage_name,
                            "stage_index": stage_index,
                            "stage_epoch": stage_epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": stage_lr,
                        }
                    )
                    fp.flush()

                    if early_stopper is not None and early_stopper.step(val_loss):
                        logger.info("Early stopping triggered for stage '%s'.", stage_name)
                        break
