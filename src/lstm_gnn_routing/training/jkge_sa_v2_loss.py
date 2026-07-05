"""Section-anomaly JKGE v2 loss.

This implements the JKGE_SA_V2 metric shared by Jawad/Farmani in a vectorized
form suitable for training batches:

    JKGE = 1 - sqrt((M + M_star + V_star + R_star) / 2)

The implementation computes the metric independently for each batch/gauge
series, then averages valid gauge scores.  It supports masks and optional gauge
weights from the curriculum trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JKGESAV2Config:
    section_length: int = 30
    eps: float = 1.0e-8
    min_valid_sections: int = 1


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0e-8)


def jkge_sa_v2_score(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    section_length: int = 30,
    eps: float = 1.0e-8,
    min_valid_sections: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute vectorized JKGE_SA_V2 scores.

    Parameters
    ----------
    preds, obs:
        Simulated and observed discharge tensors with shape ``[B, T, G]``.
    mask:
        Optional valid-value mask with shape ``[B, T, G]``. Missing values are
        excluded from section means/stds and from component averages.
    section_length:
        Length of each non-overlapping benchmark section. A value of ``999``
        uses the full available sequence as one section, matching the reference
        implementation.

    Returns
    -------
    score:
        Tensor of shape ``[B, G]`` containing one JKGE score per sample/gauge.
    components:
        Dict with tensors ``M``, ``M_star``, ``V_star``, ``R_star`` and
        ``valid_series``.  Component tensors have shape ``[B, G]``.
    """

    if preds.shape != obs.shape:
        raise ValueError(f"preds and obs must have the same shape, got {preds.shape} and {obs.shape}")
    if preds.ndim != 3:
        raise ValueError(f"JKGE_SA_V2 expects [batch,time,gauge], got shape {tuple(preds.shape)}")

    if mask is None:
        valid = torch.isfinite(preds) & torch.isfinite(obs)
    else:
        valid = mask.to(dtype=torch.bool) & torch.isfinite(preds) & torch.isfinite(obs)

    preds = torch.where(valid, preds, torch.zeros_like(preds))
    obs = torch.where(valid, obs, torch.zeros_like(obs))

    batch_size, time_steps, gauge_count = preds.shape
    if int(section_length) == 999:
        usable_steps = time_steps
        n_sections = 1
        section_length_eff = time_steps
    else:
        section_length_eff = int(section_length)
        if section_length_eff <= 0:
            raise ValueError("section_length must be positive or 999")
        n_sections = time_steps // section_length_eff
        usable_steps = n_sections * section_length_eff

    if n_sections < 1 or usable_steps < 1:
        nan_score = preds.new_full((batch_size, gauge_count), float("nan"))
        return nan_score, {
            "M": nan_score,
            "M_star": nan_score,
            "V_star": nan_score,
            "R_star": nan_score,
            "valid_series": torch.zeros_like(nan_score, dtype=torch.bool),
        }

    preds = preds[:, :usable_steps, :]
    obs = obs[:, :usable_steps, :]
    valid = valid[:, :usable_steps, :]

    # [B, S, L, G], where S is section count and L is section length.
    preds_s = preds.reshape(batch_size, n_sections, section_length_eff, gauge_count)
    obs_s = obs.reshape(batch_size, n_sections, section_length_eff, gauge_count)
    valid_s = valid.reshape(batch_size, n_sections, section_length_eff, gauge_count).to(dtype=preds.dtype)

    section_counts = valid_s.sum(dim=2)
    valid_sections = section_counts > 0.0
    valid_section_count = valid_sections.sum(dim=1)
    valid_series = valid_section_count >= int(min_valid_sections)

    pred_mean_s = (preds_s * valid_s).sum(dim=2) / section_counts.clamp_min(eps)
    obs_mean_s = (obs_s * valid_s).sum(dim=2) / section_counts.clamp_min(eps)

    pred_mean_exp = pred_mean_s.unsqueeze(2)
    obs_mean_exp = obs_mean_s.unsqueeze(2)

    pred_diff = torch.where(valid_s.bool(), preds_s - pred_mean_exp, torch.zeros_like(preds_s))
    obs_diff = torch.where(valid_s.bool(), obs_s - obs_mean_exp, torch.zeros_like(obs_s))

    pred_std_s = torch.sqrt((pred_diff.square()).sum(dim=2) / section_counts.clamp_min(eps))
    obs_std_s = torch.sqrt((obs_diff.square()).sum(dim=2) / section_counts.clamp_min(eps))

    section_weight = valid_sections.to(dtype=preds.dtype)

    # M_star: mean section-wise benchmark ratio penalty.
    m_star_sections = (1.0 - pred_mean_s / (obs_mean_s + eps)).square()
    m_star = _weighted_mean(m_star_sections, section_weight, dim=1)

    # V_star: mean section-wise standard-deviation ratio penalty.
    alpha_sections = pred_std_s / (obs_std_s + eps)
    v_star = _weighted_mean((1.0 - alpha_sections).square(), section_weight, dim=1)

    # R_star: standardized anomaly alignment using section-varying stds.
    pred_z = pred_diff / (pred_std_s.unsqueeze(2) + eps)
    obs_z = obs_diff / (obs_std_s.unsqueeze(2) + eps)
    rho = (pred_z * obs_z * valid_s).sum(dim=(1, 2)) / valid_s.sum(dim=(1, 2)).clamp_min(eps)
    r_star = (1.0 - rho).square()

    # M: global beta/mean-bias penalty.
    valid_flat = valid.to(dtype=preds.dtype)
    pred_mean = (preds * valid_flat).sum(dim=1) / valid_flat.sum(dim=1).clamp_min(eps)
    obs_mean = (obs * valid_flat).sum(dim=1) / valid_flat.sum(dim=1).clamp_min(eps)
    beta = pred_mean / (obs_mean + 1.0e-6)
    m = (beta - 1.0).square()

    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    jkge = torch.where(valid_series, jkge, torch.full_like(jkge, float("nan")))

    return jkge, {
        "M": m,
        "M_star": m_star,
        "V_star": v_star,
        "R_star": r_star,
        "valid_series": valid_series,
    }


def jkge_sa_v2_loss(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    *,
    section_length: int = 30,
    eps: float = 1.0e-8,
    min_valid_sections: int = 1,
) -> torch.Tensor:
    """Return component-averaged ``1 - JKGE_SA_V2``.

    The score helper computes components per sample/gauge.  For training we
    follow the requested multi-gauge behavior:

    1. compute M, M_star, V_star, and R_star per gauge,
    2. average each component across valid gauges, using optional gauge weights,
    3. compute one final JKGE from the averaged components,
    4. return ``1 - JKGE``.
    """

    score, components = jkge_sa_v2_score(
        preds,
        obs,
        mask,
        section_length=section_length,
        eps=eps,
        min_valid_sections=min_valid_sections,
    )
    valid = components["valid_series"]
    if weights is None:
        sample_weights = torch.ones_like(score)
    else:
        sample_weights = torch.broadcast_to(weights.to(device=score.device, dtype=score.dtype), score.shape)

    sample_weights = torch.where(valid, sample_weights, torch.zeros_like(sample_weights))
    if sample_weights.sum() <= 0:
        return score.new_tensor(0.0)

    def component_mean(name: str) -> torch.Tensor:
        value = torch.nan_to_num(components[name], nan=0.0, posinf=0.0, neginf=0.0)
        return (value * sample_weights).sum() / sample_weights.sum()

    m = component_mean("M")
    m_star = component_mean("M_star")
    v_star = component_mean("V_star")
    r_star = component_mean("R_star")
    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    return 1.0 - jkge


def jkge_sa_v1_score(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    section_length: int = 30,
    eps: float = 1.0e-8,
    min_valid_sections: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute JKGE_SA_V1 per batch/gauge using non-overlapping sections.

    SA_V1 uses section-wise means as the benchmark, but a single global anomaly
    RMS per series for the variability and correlation terms.
    """

    if preds.shape != obs.shape:
        raise ValueError(f"preds and obs must have the same shape, got {preds.shape} and {obs.shape}")
    if preds.ndim != 3:
        raise ValueError(f"JKGE_SA_V1 expects [batch,time,gauge], got shape {tuple(preds.shape)}")

    if mask is None:
        valid = torch.isfinite(preds) & torch.isfinite(obs)
    else:
        valid = mask.to(dtype=torch.bool) & torch.isfinite(preds) & torch.isfinite(obs)

    preds = torch.where(valid, preds, torch.zeros_like(preds))
    obs = torch.where(valid, obs, torch.zeros_like(obs))

    batch_size, time_steps, gauge_count = preds.shape
    if int(section_length) == 999:
        usable_steps = time_steps
        n_sections = 1
        section_length_eff = time_steps
    else:
        section_length_eff = int(section_length)
        if section_length_eff <= 0:
            raise ValueError("section_length must be positive or 999")
        n_sections = time_steps // section_length_eff
        usable_steps = n_sections * section_length_eff

    if n_sections < 1 or usable_steps < 1:
        nan_score = preds.new_full((batch_size, gauge_count), float("nan"))
        return nan_score, {
            "M": nan_score,
            "M_star": nan_score,
            "V_star": nan_score,
            "R_star": nan_score,
            "valid_series": torch.zeros_like(nan_score, dtype=torch.bool),
        }

    preds = preds[:, :usable_steps, :]
    obs = obs[:, :usable_steps, :]
    valid = valid[:, :usable_steps, :]

    preds_s = preds.reshape(batch_size, n_sections, section_length_eff, gauge_count)
    obs_s = obs.reshape(batch_size, n_sections, section_length_eff, gauge_count)
    valid_s = valid.reshape(batch_size, n_sections, section_length_eff, gauge_count).to(dtype=preds.dtype)

    section_counts = valid_s.sum(dim=2)
    valid_sections = section_counts > 0.0
    valid_section_count = valid_sections.sum(dim=1)
    valid_series = valid_section_count >= int(min_valid_sections)

    pred_mean_s = (preds_s * valid_s).sum(dim=2) / section_counts.clamp_min(eps)
    obs_mean_s = (obs_s * valid_s).sum(dim=2) / section_counts.clamp_min(eps)
    bs = pred_mean_s.unsqueeze(2).expand_as(preds_s).reshape(batch_size, usable_steps, gauge_count)
    bo = obs_mean_s.unsqueeze(2).expand_as(obs_s).reshape(batch_size, usable_steps, gauge_count)

    valid_float = valid.to(dtype=preds.dtype)
    valid_counts = valid_float.sum(dim=1)

    section_weight = valid_sections.to(dtype=preds.dtype)
    m_star_sections = (1.0 - pred_mean_s / (obs_mean_s + eps)).square()
    m_star = _weighted_mean(m_star_sections, section_weight, dim=1)

    pred_anom = torch.where(valid, preds - bs, torch.zeros_like(preds))
    obs_anom = torch.where(valid, obs - bo, torch.zeros_like(obs))
    psi_s = torch.sqrt(pred_anom.square().sum(dim=1) / valid_counts.clamp_min(eps))
    psi_o = torch.sqrt(obs_anom.square().sum(dim=1) / valid_counts.clamp_min(eps))
    alpha = psi_s / (psi_o + eps)
    v_star = (1.0 - alpha).square()

    rho = ((pred_anom / (psi_s.unsqueeze(1) + eps)) * (obs_anom / (psi_o.unsqueeze(1) + eps)) * valid_float).sum(
        dim=1
    ) / valid_counts.clamp_min(eps)
    r_star = (1.0 - rho).square()

    pred_mean = (preds * valid_float).sum(dim=1) / valid_counts.clamp_min(eps)
    obs_mean = (obs * valid_float).sum(dim=1) / valid_counts.clamp_min(eps)
    beta = pred_mean / (obs_mean + 1.0e-6)
    m = (beta - 1.0).square()

    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    jkge = torch.where(valid_series, jkge, torch.full_like(jkge, float("nan")))
    return jkge, {
        "M": m,
        "M_star": m_star,
        "V_star": v_star,
        "R_star": r_star,
        "valid_series": valid_series,
    }


def jkge_sa_v1_loss(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    *,
    section_length: int = 30,
    eps: float = 1.0e-8,
    min_valid_sections: int = 1,
) -> torch.Tensor:
    """Component-averaged ``1 - JKGE_SA_V1`` loss."""

    score, components = jkge_sa_v1_score(
        preds,
        obs,
        mask,
        section_length=section_length,
        eps=eps,
        min_valid_sections=min_valid_sections,
    )
    valid = components["valid_series"]
    if weights is None:
        sample_weights = torch.ones_like(score)
    else:
        sample_weights = torch.broadcast_to(weights.to(device=score.device, dtype=score.dtype), score.shape)

    sample_weights = torch.where(valid, sample_weights, torch.zeros_like(sample_weights))
    if sample_weights.sum() <= 0:
        return score.new_tensor(0.0)

    def component_mean(name: str) -> torch.Tensor:
        value = torch.nan_to_num(components[name], nan=0.0, posinf=0.0, neginf=0.0)
        return (value * sample_weights).sum() / sample_weights.sum()

    m = component_mean("M")
    m_star = component_mean("M_star")
    v_star = component_mean("V_star")
    r_star = component_mean("R_star")
    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    return 1.0 - jkge


def _centered_moving_average_valid(values: torch.Tensor, valid: torch.Tensor, window_length: int) -> torch.Tensor:
    """Masked centered moving average with valid convolution length."""

    batch_size, time_steps, gauge_count = values.shape
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if time_steps < window_length:
        return values.new_empty((batch_size, 0, gauge_count))

    x = values.permute(0, 2, 1).reshape(batch_size * gauge_count, 1, time_steps)
    w = valid.to(dtype=values.dtype).permute(0, 2, 1).reshape(batch_size * gauge_count, 1, time_steps)
    kernel = values.new_ones((1, 1, window_length))

    summed = F.conv1d(x * w, kernel).reshape(batch_size, gauge_count, -1).permute(0, 2, 1)
    counts = F.conv1d(w, kernel).reshape(batch_size, gauge_count, -1).permute(0, 2, 1)
    return summed / counts.clamp_min(1.0e-8)


def jkge_ma_v1_score(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    window_length: int = 31,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute JKGE_MA_V1 per batch/gauge using centered valid moving averages."""

    if preds.shape != obs.shape:
        raise ValueError(f"preds and obs must have the same shape, got {preds.shape} and {obs.shape}")
    if preds.ndim != 3:
        raise ValueError(f"JKGE_MA_V1 expects [batch,time,gauge], got shape {tuple(preds.shape)}")

    if mask is None:
        valid = torch.isfinite(preds) & torch.isfinite(obs)
    else:
        valid = mask.to(dtype=torch.bool) & torch.isfinite(preds) & torch.isfinite(obs)

    preds = torch.where(valid, preds, torch.zeros_like(preds))
    obs = torch.where(valid, obs, torch.zeros_like(obs))

    window_length = int(window_length)
    half = window_length // 2
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if preds.shape[1] < window_length:
        nan_score = preds.new_full((preds.shape[0], preds.shape[2]), float("nan"))
        return nan_score, {
            "M": nan_score,
            "M_star": nan_score,
            "V_star": nan_score,
            "R_star": nan_score,
            "valid_series": torch.zeros_like(nan_score, dtype=torch.bool),
        }

    bs = _centered_moving_average_valid(preds, valid, window_length)
    bo = _centered_moving_average_valid(obs, valid, window_length)
    preds_trim = preds[:, half : preds.shape[1] - half, :]
    obs_trim = obs[:, half : obs.shape[1] - half, :]
    valid_trim = valid[:, half : valid.shape[1] - half, :]
    if preds_trim.shape[1] != bs.shape[1]:
        # Even window lengths have ambiguous centering; keep the implementation
        # aligned with the reference odd-window usage by trimming to MA length.
        usable = min(preds_trim.shape[1], bs.shape[1])
        preds_trim = preds_trim[:, :usable, :]
        obs_trim = obs_trim[:, :usable, :]
        valid_trim = valid_trim[:, :usable, :]
        bs = bs[:, :usable, :]
        bo = bo[:, :usable, :]

    valid_float = valid_trim.to(dtype=preds.dtype)
    valid_counts = valid_float.sum(dim=1)
    valid_series = valid_counts > 2

    m_star = (((1.0 - bs / (bo + eps)).square()) * valid_float).sum(dim=1) / valid_counts.clamp_min(eps)

    pred_anom = torch.where(valid_trim, preds_trim - bs, torch.zeros_like(preds_trim))
    obs_anom = torch.where(valid_trim, obs_trim - bo, torch.zeros_like(obs_trim))
    psi_s = torch.sqrt(pred_anom.square().sum(dim=1) / valid_counts.clamp_min(eps))
    psi_o = torch.sqrt(obs_anom.square().sum(dim=1) / valid_counts.clamp_min(eps))
    alpha = psi_s / (psi_o + eps)
    v_star = (1.0 - alpha).square()

    rho = ((pred_anom / (psi_s.unsqueeze(1) + eps)) * (obs_anom / (psi_o.unsqueeze(1) + eps)) * valid_float).sum(
        dim=1
    ) / valid_counts.clamp_min(eps)
    r_star = (1.0 - rho).square()

    pred_mean = (preds_trim * valid_float).sum(dim=1) / valid_counts.clamp_min(eps)
    obs_mean = (obs_trim * valid_float).sum(dim=1) / valid_counts.clamp_min(eps)
    beta = pred_mean / (obs_mean + 1.0e-6)
    m = (beta - 1.0).square()

    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    jkge = torch.where(valid_series, jkge, torch.full_like(jkge, float("nan")))
    return jkge, {
        "M": m,
        "M_star": m_star,
        "V_star": v_star,
        "R_star": r_star,
        "valid_series": valid_series,
    }


def jkge_ma_v1_loss(
    preds: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    *,
    window_length: int = 31,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Component-averaged ``1 - JKGE_MA_V1`` loss."""

    score, components = jkge_ma_v1_score(preds, obs, mask, window_length=window_length, eps=eps)
    valid = components["valid_series"]
    if weights is None:
        sample_weights = torch.ones_like(score)
    else:
        sample_weights = torch.broadcast_to(weights.to(device=score.device, dtype=score.dtype), score.shape)

    sample_weights = torch.where(valid, sample_weights, torch.zeros_like(sample_weights))
    if sample_weights.sum() <= 0:
        return score.new_tensor(0.0)

    def component_mean(name: str) -> torch.Tensor:
        value = torch.nan_to_num(components[name], nan=0.0, posinf=0.0, neginf=0.0)
        return (value * sample_weights).sum() / sample_weights.sum()

    m = component_mean("M")
    m_star = component_mean("M_star")
    v_star = component_mean("V_star")
    r_star = component_mean("R_star")
    jkge = 1.0 - torch.sqrt((m + m_star + v_star + r_star).clamp_min(0.0) / 2.0)
    return 1.0 - jkge
