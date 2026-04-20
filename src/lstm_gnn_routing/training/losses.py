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

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


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


class KGELoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del weights
        r, alpha, beta, valid_series = _compute_kge_components(predictions, targets, mask=mask)
        kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        return 1.0 - kge[valid_series].mean()


class KGEAveragedComponentsLoss(BaseLoss):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del weights
        r, alpha, beta, valid_series = _compute_kge_components(predictions, targets, mask=mask)

        mean_r = r[valid_series].mean()
        mean_alpha = alpha[valid_series].mean()
        mean_beta = beta[valid_series].mean()
        kge = 1.0 - torch.sqrt(
            (mean_r - 1.0) ** 2
            + (mean_alpha - 1.0) ** 2
            + (mean_beta - 1.0) ** 2
        )
        return 1.0 - kge


def get_loss_function(config: Any | None = None) -> BaseLoss:
    loss_name = "masked_mse"
    if config is not None and hasattr(config, "section"):
        training_cfg = config.section("training")
        loss_name = str(training_cfg.get("loss", loss_name)).lower()

    if loss_name in {"masked_mse", "mse"}:
        return MaskedMSELoss(config)
    if loss_name in {"masked_log_mse", "log_mse", "log1p_mse"}:
        return MaskedLogMSELoss(config)
    if loss_name == "kge":
        return KGELoss(config)
    if loss_name in {"kge_averaged_components", "kge_component_mean", "kge_mean_components"}:
        return KGEAveragedComponentsLoss(config)

    raise ValueError(f"Unsupported training loss '{loss_name}'")
