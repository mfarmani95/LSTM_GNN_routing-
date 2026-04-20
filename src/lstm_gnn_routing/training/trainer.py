from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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


def _move_value(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_value(item, device) for item in value)
    return value


def _trim_time_prefix(outputs: Mapping[str, torch.Tensor], steps: int) -> dict[str, torch.Tensor]:
    if steps <= 0:
        return dict(outputs)
    trimmed = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.ndim >= 2:
            trimmed[key] = value[:, steps:].contiguous()
        else:
            trimmed[key] = value
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

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        runoff_batch = _move_value(batch, self.runoff_device)
        runoff_outputs = self.runoff_model(runoff_batch)

        pre_trim = self._context_steps(batch, "runoff_pre_routing_trim_steps")
        runoff_outputs = _trim_time_prefix(runoff_outputs, pre_trim)

        routing_batch = _move_value(batch, self.routing_device)
        runoff_outputs = {
            key: value.to(device=self.routing_device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in runoff_outputs.items()
        }
        if self.runoff_transfer is not None:
            runoff_outputs = self.runoff_transfer(runoff_outputs, routing_batch)

        prediction = self.routing_model(runoff_outputs, routing_batch)
        routing_context = self._context_steps(batch, "routing_context_steps")
        if routing_context > 0:
            prediction = prediction[:, routing_context:].contiguous()
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
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = training_cfg.get("grad_clip_norm")
        self.skip_nonfinite_batches = bool(training_cfg.get("skip_nonfinite_batches", True))
        self.show_progress = bool(training_cfg.get("show_progress", True))
        self.leave_progress = bool(training_cfg.get("leave_progress", False))
        self.prediction_key = str(training_cfg.get("default_prediction_key", "runoff_total"))

        self.optimizer = self._build_optimizer(float(training_cfg.get("learning_rate", 1e-3)))
        self.history_path = self.run_dir / "training_history.csv"

    def _build_optimizer(self, learning_rate: float):
        training_cfg = self.config.section("training")
        optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
        params = [param for param in self.model.parameters() if param.requires_grad]
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=learning_rate)
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=learning_rate)
        raise ValueError("training.optimizer must be one of: adam, adamw")

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
                with autocast(enabled=self.use_amp):
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
