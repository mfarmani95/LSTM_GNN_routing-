from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = mask
    while result.ndim < target.ndim:
        result = result.unsqueeze(-1)
    return result.expand_as(target)


def _broadcast_weights(weights: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
    if weights is None:
        return None
    result = weights.to(dtype=target.dtype, device=target.device)
    if result.ndim == 1 and target.ndim >= 2:
        view_shape = [1] * target.ndim
        view_shape[-1] = int(result.shape[0])
        result = result.reshape(*view_shape)
        return result.expand_as(target)
    while result.ndim < target.ndim:
        result = result.unsqueeze(-1)
    return result.expand_as(target)


def _combine_valid_mask(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = torch.isfinite(predictions) & torch.isfinite(targets)
    if mask is not None:
        valid = valid & _broadcast_mask(mask.to(dtype=torch.bool, device=targets.device), targets)
    return valid


def _reshape_series_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, tensor.shape[0], 1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(-1)
    return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


def _compute_kge_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = _reshape_series_first(predictions)
    obs = _reshape_series_first(targets)
    valid = _combine_valid_mask(preds, obs, mask=mask)

    eps = torch.finfo(predictions.dtype).eps
    valid_counts = valid.sum(dim=1)
    valid_series = valid_counts > 1
    if not torch.any(valid_series):
        raise ValueError("KGE loss received fewer than two valid points for every gauge series.")
    valid_counts = valid_counts.clamp(min=1).to(predictions.dtype)

    preds = torch.where(valid, preds, torch.zeros_like(preds))
    obs = torch.where(valid, obs, torch.zeros_like(obs))

    mu_s = preds.sum(dim=1) / valid_counts
    mu_o = obs.sum(dim=1) / valid_counts

    sim_anom = torch.where(valid, preds - mu_s.unsqueeze(1), torch.zeros_like(preds))
    obs_anom = torch.where(valid, obs - mu_o.unsqueeze(1), torch.zeros_like(obs))

    std_s = torch.sqrt((sim_anom**2).sum(dim=1) / valid_counts + eps)
    std_o = torch.sqrt((obs_anom**2).sum(dim=1) / valid_counts + eps)

    r_num = (sim_anom * obs_anom).sum(dim=1)
    r_den = torch.sqrt((sim_anom**2).sum(dim=1) * (obs_anom**2).sum(dim=1) + eps)
    r = r_num / (r_den + eps)

    alpha = std_s / (std_o + eps)
    beta = (mu_s + eps) / (mu_o + eps)
    return r, alpha, beta, valid_series


class BaseLoss(nn.Module):
    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config
        training_cfg = config.section("training") if config is not None and hasattr(config, "section") else {}
        self.target_space = str(
            training_cfg.get("loss_target_space", training_cfg.get("kge_target_space", "normalized"))
        ).lower()
        self.target_transform = "identity"
        self._warned_missing_target_scaler = False
        self.register_buffer("_target_means", torch.empty(0), persistent=False)
        self.register_buffer("_target_stds", torch.empty(0), persistent=False)

    def configure_target_scaler(self, scaler: dict[str, Any] | None, *, device: torch.device | None = None) -> None:
        if not scaler:
            return
        means = torch.as_tensor(scaler.get("means"), dtype=torch.float32, device=device)
        stds = torch.as_tensor(scaler.get("stds"), dtype=torch.float32, device=device)
        if means.ndim == 2 and means.shape[1] == 1:
            means = means[:, 0]
            stds = stds[:, 0]
        elif means.ndim != 1:
            logger.warning(
                "Loss target inverse transform supports a single target variable per gauge; "
                "received scaler shape %s. Loss will stay in normalized units.",
                tuple(int(v) for v in means.shape),
            )
            return
        self.target_transform = str(scaler.get("transform", "identity")).lower()
        self._target_means = means
        self._target_stds = stds.clamp_min(1.0e-6)

    def _maybe_inverse_transform_targets(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_space not in {"physical", "original", "unscaled"}:
            return predictions, targets
        if int(self._target_means.numel()) == 0 or int(self._target_stds.numel()) == 0:
            if not self._warned_missing_target_scaler:
                logger.warning("KGE loss requested physical target space but no target scaler was configured.")
                self._warned_missing_target_scaler = True
            return predictions, targets
        if predictions.ndim != 3 or targets.ndim != 3:
            raise ValueError(
                "Physical-space loss inverse transform expects [B,T,G] tensors, "
                f"got prediction={tuple(predictions.shape)} target={tuple(targets.shape)}"
            )
        if int(predictions.shape[-1]) != int(self._target_means.numel()):
            raise ValueError(
                f"Target scaler has {int(self._target_means.numel())} gauges, "
                f"but loss tensor shape is {tuple(predictions.shape)}"
            )
        means = self._target_means.to(device=predictions.device, dtype=predictions.dtype).view(1, 1, -1)
        stds = self._target_stds.to(device=predictions.device, dtype=predictions.dtype).view(1, 1, -1)
        pred_values = predictions * stds + means
        target_values = targets * stds + means
        if self.target_transform in {"", "none", "identity"}:
            return pred_values, target_values
        if self.target_transform == "log1p":
            return torch.expm1(pred_values), torch.expm1(target_values)
        raise ValueError(f"Unsupported target transform '{self.target_transform}' for physical-space loss.")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _series_weights(
    weights: torch.Tensor | None,
    obs: torch.Tensor,
    valid: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    if weights is None:
        return torch.ones_like(valid_counts, dtype=obs.dtype, device=obs.device)
    weight_tensor = _broadcast_weights(weights, obs)
    if weight_tensor is None:
        return torch.ones_like(valid_counts, dtype=obs.dtype, device=obs.device)
    safe_weights = torch.where(valid, weight_tensor, torch.zeros_like(weight_tensor))
    return safe_weights.sum(dim=1) / valid_counts.clamp(min=1).to(dtype=obs.dtype)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    safe_weights = weights.to(device=values.device, dtype=values.dtype).clamp_min(torch.finfo(values.dtype).eps)
    return (values * safe_weights).sum() / safe_weights.sum()


def _masked_moving_average(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    window: int,
) -> torch.Tensor:
    """Apply a centered masked moving average along the time axis."""

    if window <= 1:
        return values
    batch_size, time_steps, series_count = values.shape
    flat_values = values.permute(0, 2, 1).reshape(batch_size * series_count, 1, time_steps)
    flat_valid = valid.to(dtype=values.dtype).permute(0, 2, 1).reshape(batch_size * series_count, 1, time_steps)
    kernel = torch.ones(1, 1, window, dtype=values.dtype, device=values.device)
    left_padding = (window - 1) // 2
    right_padding = window // 2
    padded_values = torch.nn.functional.pad(flat_values * flat_valid, (left_padding, right_padding))
    padded_valid = torch.nn.functional.pad(flat_valid, (left_padding, right_padding))
    numerator = torch.nn.functional.conv1d(padded_values, kernel)
    denominator = torch.nn.functional.conv1d(padded_valid, kernel).clamp_min(torch.finfo(values.dtype).eps)
    averaged = numerator / denominator
    return averaged.reshape(batch_size, series_count, time_steps).permute(0, 2, 1)


def _masked_section_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    section_length: int,
) -> torch.Tensor:
    """Piecewise constant masked benchmark along the time axis."""

    if section_length <= 1:
        return values
    result = torch.zeros_like(values)
    valid_float = valid.to(dtype=values.dtype)
    for start in range(0, int(values.shape[1]), int(section_length)):
        stop = min(start + int(section_length), int(values.shape[1]))
        section_valid = valid_float[:, start:stop]
        denominator = section_valid.sum(dim=1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        section_mean = (values[:, start:stop] * section_valid).sum(dim=1, keepdim=True) / denominator
        result[:, start:stop] = section_mean
    return result


class MaskedMSELoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = _reshape_series_first(predictions)
        obs = _reshape_series_first(targets)
        valid = _combine_valid_mask(preds, obs, mask=mask)
        valid_counts = valid.sum(dim=1)
        valid_series = valid_counts > 0
        if not torch.any(valid_series):
            finite_predictions = int(torch.isfinite(predictions).sum().detach().cpu().item())
            finite_targets = int(torch.isfinite(targets).sum().detach().cpu().item())
            explicit_mask_valid = (
                int(_broadcast_mask(mask.to(dtype=torch.bool, device=targets.device), obs).sum().detach().cpu().item())
                if mask is not None
                else "not provided"
            )
            raise ValueError(
                "MaskedMSELoss received no valid target points. "
                f"finite_predictions={finite_predictions}, finite_targets={finite_targets}, "
                f"mask_valid={explicit_mask_valid}, prediction_shape={tuple(predictions.shape)}, "
                f"target_shape={tuple(targets.shape)}"
            )

        safe_preds = torch.where(valid, preds, torch.zeros_like(preds))
        safe_obs = torch.where(valid, obs, torch.zeros_like(obs))
        weight_tensor = _broadcast_weights(weights, obs)
        if weight_tensor is None:
            weight_tensor = torch.ones_like(obs)
        weight_tensor = torch.where(valid, weight_tensor, torch.zeros_like(weight_tensor))
        error = (safe_preds - safe_obs) ** 2
        weighted_error_sum = (error * weight_tensor).sum(dim=1)
        series_weight_sum = weight_tensor.sum(dim=1).clamp(min=torch.finfo(predictions.dtype).eps)
        series_mse = weighted_error_sum / series_weight_sum
        series_weights = series_weight_sum / valid_counts.clamp(min=1).to(predictions.dtype)
        valid_series_weights = series_weights[valid_series].clamp(min=torch.finfo(predictions.dtype).eps)
        return (series_mse[valid_series] * valid_series_weights).sum() / valid_series_weights.sum()


class MaskedLogMSELoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_preds = _reshape_series_first(predictions)
        raw_obs = _reshape_series_first(targets)
        valid = _combine_valid_mask(raw_preds, raw_obs, mask=mask)
        valid_counts = valid.sum(dim=1)
        valid_series = valid_counts > 0
        if not torch.any(valid_series):
            raise ValueError("MaskedLogMSELoss received no valid finite prediction/target points.")

        safe_preds = torch.where(valid, raw_preds, torch.zeros_like(raw_preds))
        safe_obs = torch.where(valid, raw_obs, torch.zeros_like(raw_obs))
        preds = torch.log1p(torch.clamp(safe_preds, min=0.0))
        obs = torch.log1p(torch.clamp(safe_obs, min=0.0))
        weight_tensor = _broadcast_weights(weights, obs)
        if weight_tensor is None:
            weight_tensor = torch.ones_like(obs)
        weight_tensor = torch.where(valid, weight_tensor, torch.zeros_like(weight_tensor))
        error = (preds - obs) ** 2
        weighted_error_sum = (error * weight_tensor).sum(dim=1)
        series_weight_sum = weight_tensor.sum(dim=1).clamp(min=torch.finfo(predictions.dtype).eps)
        series_mse = weighted_error_sum / series_weight_sum
        series_weights = series_weight_sum / valid_counts.clamp(min=1).to(predictions.dtype)
        valid_series_weights = series_weights[valid_series].clamp(min=torch.finfo(predictions.dtype).eps)
        return (series_mse[valid_series] * valid_series_weights).sum() / valid_series_weights.sum()


class PeakWeightedMSELoss(BaseLoss):
    """MSE that upweights high-flow target days to discourage flat hydrographs."""

    def __init__(self, config: Any | None = None):
        super().__init__(config)
        training_cfg = config.section("training") if config is not None and hasattr(config, "section") else {}
        self.base_loss = MaskedMSELoss(config)
        self.peak_weight = float(training_cfg.get("peak_flow_weight", 2.0))
        self.peak_power = float(training_cfg.get("peak_flow_power", 1.0))
        self.peak_threshold = float(training_cfg.get("peak_flow_threshold", 0.5))
        self.max_weight = float(training_cfg.get("peak_flow_max_weight", 8.0))
        if self.peak_weight < 0.0:
            raise ValueError("training.peak_flow_weight must be non-negative.")
        if self.peak_power <= 0.0:
            raise ValueError("training.peak_flow_power must be positive.")
        if self.max_weight <= 0.0:
            raise ValueError("training.peak_flow_max_weight must be positive.")

    def configure_target_scaler(self, scaler: dict[str, Any] | None, *, device: torch.device | None = None) -> None:
        super().configure_target_scaler(scaler, device=device)
        if hasattr(self.base_loss, "configure_target_scaler"):
            self.base_loss.configure_target_scaler(scaler, device=device)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = _reshape_series_first(targets)
        peak_signal = torch.relu(obs - self.peak_threshold)
        peak_weights = 1.0 + self.peak_weight * torch.pow(peak_signal, self.peak_power)
        peak_weights = peak_weights.clamp(max=self.max_weight)
        base_weights = _broadcast_weights(weights, obs)
        if base_weights is not None:
            peak_weights = peak_weights * base_weights
        return self.base_loss(predictions, targets, mask=mask, weights=peak_weights)


class KGELoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = _reshape_series_first(predictions)
        obs = _reshape_series_first(targets)
        preds, obs = self._maybe_inverse_transform_targets(preds, obs)
        valid = _combine_valid_mask(preds, obs, mask=mask)
        valid_counts = valid.sum(dim=1)
        series_weights = _series_weights(weights, obs, valid, valid_counts)
        r, alpha, beta, valid_series = _compute_kge_components(preds, obs, mask=valid)
        kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        return 1.0 - _weighted_mean(kge[valid_series], series_weights[valid_series])


class KGEAveragedComponentsLoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = _reshape_series_first(predictions)
        obs = _reshape_series_first(targets)
        preds, obs = self._maybe_inverse_transform_targets(preds, obs)
        valid = _combine_valid_mask(preds, obs, mask=mask)
        valid_counts = valid.sum(dim=1)
        series_weights = _series_weights(weights, obs, valid, valid_counts)
        r, alpha, beta, valid_series = _compute_kge_components(preds, obs, mask=valid)

        valid_weights = series_weights[valid_series]
        mean_r = _weighted_mean(r[valid_series], valid_weights)
        mean_alpha = _weighted_mean(alpha[valid_series], valid_weights)
        mean_beta = _weighted_mean(beta[valid_series], valid_weights)
        kge = 1.0 - torch.sqrt(
            (mean_r - 1.0) ** 2
            + (mean_alpha - 1.0) ** 2
            + (mean_beta - 1.0) ** 2
        )
        return 1.0 - kge


class JKGELoss(BaseLoss):
    """Jawad-Kling-Gupta Efficiency loss with averaged per-location components."""

    def __init__(self, config: Any | None = None):
        super().__init__(config)
        training_cfg = config.section("training") if config is not None and hasattr(config, "section") else {}
        self.benchmark = str(
            training_cfg.get("jkge_benchmark", training_cfg.get("jkge_benchmark_operator", "moving_average"))
        ).lower()
        self.window = int(training_cfg.get("jkge_window", training_cfg.get("jkge_moving_average_window", 30)))
        self.section_length = int(training_cfg.get("jkge_section_length", self.window))
        self.eps = float(training_cfg.get("jkge_eps", 1.0e-8))
        if self.window <= 0:
            raise ValueError("training.jkge_window must be positive.")
        if self.section_length <= 0:
            raise ValueError("training.jkge_section_length must be positive.")
        if self.eps <= 0.0:
            raise ValueError("training.jkge_eps must be positive.")
        if self.benchmark not in {"moving_average", "section_mean"}:
            raise ValueError(
                "training.jkge_benchmark must be one of: moving_average, section_mean"
            )

    def _benchmark(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        if self.benchmark == "moving_average":
            return _masked_moving_average(safe_values, valid, window=self.window)
        if self.benchmark == "section_mean":
            return _masked_section_mean(safe_values, valid, section_length=self.section_length)
        raise ValueError(f"Unsupported JKGE benchmark operator '{self.benchmark}'.")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = _reshape_series_first(predictions)
        obs = _reshape_series_first(targets)
        preds, obs = self._maybe_inverse_transform_targets(preds, obs)
        valid = _combine_valid_mask(preds, obs, mask=mask)
        valid_counts = valid.sum(dim=1)
        valid_series = valid_counts > 2
        if not torch.any(valid_series):
            raise ValueError("JKGE loss received fewer than three valid points for every gauge series.")

        safe_preds = torch.where(valid, preds, torch.zeros_like(preds))
        safe_obs = torch.where(valid, obs, torch.zeros_like(obs))
        benchmark_pred = self._benchmark(safe_preds, valid)
        benchmark_obs = self._benchmark(safe_obs, valid)
        anomaly_pred = torch.where(valid, safe_preds - benchmark_pred, torch.zeros_like(preds))
        anomaly_obs = torch.where(valid, safe_obs - benchmark_obs, torch.zeros_like(obs))

        eps = torch.as_tensor(self.eps, dtype=preds.dtype, device=preds.device)
        valid_float = valid.to(dtype=preds.dtype)
        count = valid_counts.clamp(min=1).to(dtype=preds.dtype)
        observed_mean = (safe_obs * valid_float).sum(dim=1) / count
        centered_benchmark_obs = torch.where(
            valid,
            benchmark_obs - observed_mean.unsqueeze(1),
            torch.zeros_like(benchmark_obs),
        )
        benchmark_difference = torch.where(valid, benchmark_pred - benchmark_obs, torch.zeros_like(benchmark_obs))

        pred_anomaly_norm = torch.linalg.vector_norm(anomaly_pred, dim=1)
        obs_anomaly_norm = torch.linalg.vector_norm(anomaly_obs, dim=1)
        anomaly_dot = (anomaly_pred * anomaly_obs).sum(dim=1)
        correlation = anomaly_dot / (pred_anomaly_norm * obs_anomaly_norm + eps)
        variability = pred_anomaly_norm / (obs_anomaly_norm + eps)
        benchmark_mismatch = torch.linalg.vector_norm(benchmark_difference, dim=1) / (
            torch.linalg.vector_norm(centered_benchmark_obs, dim=1) + eps
        )

        finite_components = (
            torch.isfinite(correlation)
            & torch.isfinite(variability)
            & torch.isfinite(benchmark_mismatch)
            & (obs_anomaly_norm > eps)
            & valid_series
        )
        if not torch.any(finite_components):
            raise ValueError("JKGE loss received no finite valid component series.")

        series_weights = _series_weights(weights, obs, valid, valid_counts)
        valid_weights = series_weights[finite_components]
        mean_correlation = _weighted_mean(correlation[finite_components], valid_weights)
        mean_variability = _weighted_mean(variability[finite_components], valid_weights)
        mean_mismatch = _weighted_mean(benchmark_mismatch[finite_components], valid_weights)
        # Since JKGE = 1 - distance, the requested loss 1 - JKGE is the distance.
        return torch.sqrt(
            (mean_correlation - 1.0) ** 2
            + (mean_variability - 1.0) ** 2
            + mean_mismatch**2
        )


class MixedMSEKGELoss(BaseLoss):
    """Blend stable pointwise MSE with a small KGE-shaped hydrologic objective."""

    def __init__(self, config: Any | None = None):
        super().__init__(config)
        training_cfg = config.section("training") if config is not None and hasattr(config, "section") else {}
        self.mse_weight = float(training_cfg.get("mse_loss_weight", training_cfg.get("mse_weight", 0.9)))
        self.kge_weight = float(training_cfg.get("kge_loss_weight", training_cfg.get("kge_weight", 0.1)))
        if self.mse_weight < 0.0 or self.kge_weight < 0.0:
            raise ValueError("MixedMSEKGELoss weights must be non-negative.")
        if self.mse_weight == 0.0 and self.kge_weight == 0.0:
            raise ValueError("MixedMSEKGELoss requires at least one positive loss weight.")

        kge_variant = str(training_cfg.get("mixed_kge_variant", "averaged_components")).lower()
        self.mse_loss = MaskedMSELoss(config)
        if kge_variant in {"standard", "kge"}:
            self.kge_loss = KGELoss(config)
        elif kge_variant in {"averaged_components", "component_mean", "kge_averaged_components"}:
            self.kge_loss = KGEAveragedComponentsLoss(config)
        else:
            raise ValueError(
                "training.mixed_kge_variant must be one of: "
                "standard, kge, averaged_components, component_mean, kge_averaged_components"
            )

    def configure_target_scaler(self, scaler: dict[str, Any] | None, *, device: torch.device | None = None) -> None:
        super().configure_target_scaler(scaler, device=device)
        if hasattr(self.mse_loss, "configure_target_scaler"):
            self.mse_loss.configure_target_scaler(scaler, device=device)
        if hasattr(self.kge_loss, "configure_target_scaler"):
            self.kge_loss.configure_target_scaler(scaler, device=device)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_weight = self.mse_weight + self.kge_weight
        mse = self.mse_loss(predictions, targets, mask=mask, weights=weights)
        kge = self.kge_loss(predictions, targets, mask=mask, weights=weights)
        return (self.mse_weight * mse + self.kge_weight * kge) / total_weight


def get_loss_function(config: Any | None = None) -> BaseLoss:
    loss_name = "masked_mse"
    if config is not None and hasattr(config, "section"):
        training_cfg = config.section("training")
        loss_name = str(training_cfg.get("loss", loss_name)).lower()

    if loss_name in {"masked_mse", "mse"}:
        return MaskedMSELoss(config)
    if loss_name in {"masked_log_mse", "log_mse", "log1p_mse"}:
        return MaskedLogMSELoss(config)
    if loss_name in {"peak_weighted_mse", "high_flow_weighted_mse", "flow_weighted_mse"}:
        return PeakWeightedMSELoss(config)
    if loss_name == "kge":
        return KGELoss(config)
    if loss_name in {"kge_averaged_components", "kge_component_mean", "kge_mean_components", "component_mean_kge"}:
        return KGEAveragedComponentsLoss(config)
    if loss_name in {"jkge", "jawad_kge", "jawad_kling_gupta", "jawad_kling_gupta_efficiency"}:
        return JKGELoss(config)
    if loss_name in {"mixed_mse_kge", "mse_kge", "masked_mse_kge"}:
        return MixedMSEKGELoss(config)

    raise ValueError(f"Unsupported training loss '{loss_name}'")
