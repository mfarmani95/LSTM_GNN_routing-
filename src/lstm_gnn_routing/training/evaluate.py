from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from lstm_gnn_routing.dataset.batcher import RoutingBatcher
from lstm_gnn_routing.dataset.dataset import RoutingDataset
from lstm_gnn_routing.training.model_factory import (
    build_routing_model,
    build_runoff_model,
    build_runoff_transfer_model,
)
from lstm_gnn_routing.training.trainer import RunoffRoutingPipeline, _PerGaugeValidationMetrics
from lstm_gnn_routing.utils.config import RoutingConfig

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved routing run on validation/test data.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing config.yml and checkpoints")
    parser.add_argument("--period", default="test", choices=("validation", "test"), help="Dataset split to evaluate")
    parser.add_argument(
        "--checkpoint-file",
        default="best_final_stage_model.pt",
        help="Checkpoint filename relative to run-dir, or an absolute checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for evaluation outputs. Defaults to <run-dir>/evaluation_<period>_<checkpoint-stem>",
    )
    parser.add_argument(
        "--noah-config",
        type=str,
        default=None,
        help="Optional Noah config for noah_table_priors static features",
    )
    return parser.parse_args(argv)


def _resolve_checkpoint_path(run_dir: Path, checkpoint_file: str) -> Path:
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return run_dir / checkpoint_path


def _default_output_dir(run_dir: Path, period: str, checkpoint_path: Path) -> Path:
    return run_dir / f"evaluation_{period}_{checkpoint_path.stem}"


def _resolve_device(preferred: Any, *, fallback: str = "cpu") -> torch.device:
    text = str(preferred or fallback)
    if text.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested device '%s' but CUDA is unavailable; falling back to CPU.", text)
        return torch.device("cpu")
    return torch.device(text)


def _prepare_eval_tensors(batch: Mapping[str, Any], prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch["y"].to(device=prediction.device, non_blocking=True)
    target_mask = batch["target_mask"].to(device=prediction.device, non_blocking=True)
    if target.ndim == 4 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if target_mask.ndim == 4 and target_mask.shape[-1] == 1:
        target_mask = target_mask.squeeze(-1)

    if prediction.shape[1] != target.shape[1]:
        common = min(int(prediction.shape[1]), int(target.shape[1]))
        prediction = prediction[:, -common:]
        target = target[:, -common:]
        target_mask = target_mask[:, -common:]

    loss_time_mask = batch["loss_mask"].to(device=prediction.device, non_blocking=True)
    if loss_time_mask.ndim == 2:
        loss_time_mask = loss_time_mask[:, -target.shape[1] :].unsqueeze(-1).expand_as(target_mask)
    valid_mask = target_mask.to(dtype=torch.bool) & loss_time_mask.to(dtype=torch.bool)
    return prediction, target, valid_mask


def _build_target_inverse_transform(dataset: RoutingDataset, *, device: torch.device):
    scaler = getattr(dataset, "target_scaler", None)
    if not scaler:
        return None
    means = np.asarray(scaler.get("means"), dtype=np.float32)
    stds = np.asarray(scaler.get("stds"), dtype=np.float32)
    if means.ndim == 2 and means.shape[1] == 1:
        means = means[:, 0]
        stds = stds[:, 0]
    elif means.ndim != 1:
        logger.warning(
            "Evaluation inverse transform supports one target variable per gauge; received scaler shape %s.",
            tuple(int(v) for v in means.shape),
        )
        return None

    transform = str(scaler.get("transform", "identity")).lower()
    mean_tensor = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, 1, -1)
    std_tensor = torch.as_tensor(stds, dtype=torch.float32, device=device).view(1, 1, -1)

    def inverse(values: torch.Tensor) -> torch.Tensor:
        result = values * std_tensor.to(device=values.device, dtype=values.dtype) + mean_tensor.to(
            device=values.device,
            dtype=values.dtype,
        )
        if transform in {"", "none", "identity"}:
            return result
        if transform == "log1p":
            return torch.expm1(result)
        raise ValueError(f"Unsupported target transform '{transform}' during evaluation.")

    return inverse


def _to_iso_date(value: Any) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def _safe_float(value: float) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _compute_series_metrics(predictions: np.ndarray, observations: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(predictions) & np.isfinite(observations)
    preds = predictions[valid]
    obs = observations[valid]
    n = int(preds.size)
    if n == 0:
        return {
            "valid_points": 0,
            "mse": None,
            "rmse": None,
            "kge": None,
            "nse": None,
            "correlation": None,
            "alpha": None,
            "beta": None,
        }

    mse = float(np.mean((preds - obs) ** 2))
    rmse = float(np.sqrt(mse))
    mean_pred = float(np.mean(preds))
    mean_obs = float(np.mean(obs))
    std_pred = float(np.std(preds))
    std_obs = float(np.std(obs))
    sst = float(np.sum((obs - mean_obs) ** 2))
    eps = float(np.finfo(np.float64).eps)

    if n > 1 and std_pred > eps and std_obs > eps:
        correlation = float(np.corrcoef(preds, obs)[0, 1])
    else:
        correlation = None

    alpha = (std_pred / std_obs) if std_obs > eps else None
    beta = (mean_pred / mean_obs) if abs(mean_obs) > eps else None
    if correlation is None or alpha is None or beta is None:
        kge = None
    else:
        kge = float(1.0 - math.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    nse = None if sst <= eps else float(1.0 - np.sum((preds - obs) ** 2) / sst)
    return {
        "valid_points": n,
        "mse": mse,
        "rmse": rmse,
        "kge": kge,
        "nse": nse,
        "correlation": correlation,
        "alpha": alpha,
        "beta": beta,
    }


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_run(
    *,
    run_dir: Path,
    period: str,
    checkpoint_path: Path,
    output_dir: Path,
    noah_config: str | None = None,
) -> dict[str, Any]:
    config_path = run_dir / "config.yml"
    config = RoutingConfig.from_yaml(config_path)
    if noah_config is not None:
        config.set_noah_config_path(noah_config)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RoutingDataset(config, period)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=RoutingBatcher.collate_fn,
        pin_memory=False,
    )
    example_batch = next(iter(loader))

    execution_cfg = config.section("execution")
    default_device = _resolve_device(config.device)
    runoff_device = _resolve_device(execution_cfg.get("runoff_model_device", default_device))
    routing_device = _resolve_device(execution_cfg.get("routing_device", default_device))

    runoff_model = build_runoff_model(config, example_batch=example_batch, device=runoff_device)
    runoff_transfer = build_runoff_transfer_model(config, example_batch=example_batch, device=routing_device)
    routing_model = build_routing_model(config, example_batch=example_batch, device=routing_device)
    model = RunoffRoutingPipeline(
        runoff_model=runoff_model,
        runoff_transfer=runoff_transfer,
        routing_model=routing_model,
        runoff_device=runoff_device,
        routing_device=routing_device,
        prediction_key=str(config.section("training").get("default_prediction_key", "runoff_total")),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inverse_transform = _build_target_inverse_transform(dataset, device=routing_device)
    window_metrics = _PerGaugeValidationMetrics()
    aggregated_daily: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {
            "prediction_sum": 0.0,
            "prediction_count": 0,
            "observation_sum": 0.0,
            "observation_count": 0,
            "metric_count": 0,
        }
    )
    gauge_order: list[str] | None = None

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            prediction, target, mask = _prepare_eval_tensors(batch, outputs[model.prediction_key])
            prediction = prediction.detach()
            target = target.detach()
            if inverse_transform is not None:
                metric_prediction = inverse_transform(prediction)
                metric_target = inverse_transform(target)
            else:
                metric_prediction = prediction
                metric_target = target
            window_metrics.update(metric_prediction, metric_target, mask)

            prediction_np = metric_prediction.cpu().numpy()
            target_np = metric_target.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            batch_size = int(prediction_np.shape[0])
            common = int(prediction_np.shape[1])

            x_info_batch = batch.get("x_info", [])
            for batch_index in range(batch_size):
                info = x_info_batch[batch_index]
                gauge_ids = [str(value) for value in info.get("catchment_ids", [])]
                if gauge_order is None:
                    gauge_order = gauge_ids
                dates = np.asarray(info.get("target_time_index"))[-common:]
                for time_index, raw_date in enumerate(dates):
                    date_text = _to_iso_date(raw_date)
                    for gauge_index, gauge_id in enumerate(gauge_ids):
                        key = (gauge_id, date_text)
                        pred_value = float(prediction_np[batch_index, time_index, gauge_index])
                        if math.isfinite(pred_value):
                            aggregated_daily[key]["prediction_sum"] += pred_value
                            aggregated_daily[key]["prediction_count"] += 1
                        if bool(mask_np[batch_index, time_index, gauge_index]):
                            obs_value = float(target_np[batch_index, time_index, gauge_index])
                            if math.isfinite(obs_value):
                                aggregated_daily[key]["observation_sum"] += obs_value
                                aggregated_daily[key]["observation_count"] += 1
                                aggregated_daily[key]["metric_count"] += 1

    if gauge_order is None:
        raise RuntimeError(f"No samples were produced for period '{period}'.")

    daily_rows: list[dict[str, Any]] = []
    series_by_gauge: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for gauge_id, date_text in sorted(aggregated_daily, key=lambda value: (value[0], value[1])):
        record = aggregated_daily[(gauge_id, date_text)]
        prediction = (
            record["prediction_sum"] / record["prediction_count"]
            if int(record["prediction_count"]) > 0
            else float("nan")
        )
        observation = (
            record["observation_sum"] / record["observation_count"]
            if int(record["observation_count"]) > 0
            else float("nan")
        )
        daily_rows.append(
            {
                "gauge_id": gauge_id,
                "date": date_text,
                "prediction": _safe_float(float(prediction)),
                "observation": _safe_float(float(observation)),
                "prediction_windows": int(record["prediction_count"]),
                "observation_windows": int(record["observation_count"]),
                "metric_windows": int(record["metric_count"]),
            }
        )
        if math.isfinite(float(prediction)) and math.isfinite(float(observation)):
            series_by_gauge[gauge_id].append((date_text, float(prediction), float(observation)))

    window_summary = window_metrics.summarize()
    window_by_gauge = {
        str(row["gauge_id"]): row for row in window_summary.get("per_gauge_metrics", [])
    }

    daily_metrics_by_gauge: list[dict[str, Any]] = []
    for gauge_id in gauge_order:
        series = sorted(series_by_gauge.get(gauge_id, []), key=lambda row: row[0])
        if series:
            preds = np.asarray([item[1] for item in series], dtype=np.float64)
            obs = np.asarray([item[2] for item in series], dtype=np.float64)
        else:
            preds = np.asarray([], dtype=np.float64)
            obs = np.asarray([], dtype=np.float64)
        daily_metrics = _compute_series_metrics(preds, obs)
        window_metrics_row = window_by_gauge.get(gauge_id, {})
        daily_metrics_by_gauge.append(
            {
                "gauge_id": gauge_id,
                "window_kge": _safe_float(float(window_metrics_row["kge"])) if window_metrics_row.get("kge") is not None else None,
                "window_nse": _safe_float(float(window_metrics_row["nse"])) if window_metrics_row.get("nse") is not None else None,
                "window_mse": _safe_float(float(window_metrics_row["mse"])) if window_metrics_row.get("mse") is not None else None,
                "window_valid_points": int(window_metrics_row.get("valid_points", 0) or 0),
                "daily_kge": _safe_float(daily_metrics["kge"]),
                "daily_nse": _safe_float(daily_metrics["nse"]),
                "daily_mse": _safe_float(daily_metrics["mse"]),
                "daily_rmse": _safe_float(daily_metrics["rmse"]),
                "daily_valid_days": int(daily_metrics["valid_points"]),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "config_file": str(config_path),
        "checkpoint_file": str(checkpoint_path),
        "period": period,
        "window_metrics": {
            "mean_kge": _safe_float(float(window_summary["mean_kge"])) if window_summary.get("mean_kge") == window_summary.get("mean_kge") else None,
            "mean_nse": _safe_float(float(window_summary["mean_nse"])) if window_summary.get("mean_nse") == window_summary.get("mean_nse") else None,
            "valid_kge_gauges": int(window_summary.get("valid_kge_gauges", 0) or 0),
            "valid_nse_gauges": int(window_summary.get("valid_nse_gauges", 0) or 0),
        },
        "daily_metrics": {
            "mean_kge": _mean_or_none(row["daily_kge"] for row in daily_metrics_by_gauge),
            "mean_nse": _mean_or_none(row["daily_nse"] for row in daily_metrics_by_gauge),
            "mean_rmse": _mean_or_none(row["daily_rmse"] for row in daily_metrics_by_gauge),
            "valid_kge_gauges": sum(1 for row in daily_metrics_by_gauge if row["daily_kge"] is not None),
            "valid_nse_gauges": sum(1 for row in daily_metrics_by_gauge if row["daily_nse"] is not None),
            "valid_rmse_gauges": sum(1 for row in daily_metrics_by_gauge if row["daily_rmse"] is not None),
            "timeseries_rows": len(daily_rows),
        },
    }

    _write_csv(
        output_dir / f"{period}_timeseries.csv",
        ["gauge_id", "date", "prediction", "observation", "prediction_windows", "observation_windows", "metric_windows"],
        daily_rows,
    )
    _write_csv(
        output_dir / f"{period}_metrics_by_gauge.csv",
        [
            "gauge_id",
            "window_kge",
            "window_nse",
            "window_mse",
            "window_valid_points",
            "daily_kge",
            "daily_nse",
            "daily_mse",
            "daily_rmse",
            "daily_valid_days",
        ],
        daily_metrics_by_gauge,
    )
    with (output_dir / f"{period}_metrics_summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved evaluation outputs to %s", output_dir)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    run_dir = args.run_dir.resolve()
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint_file).resolve()
    output_dir = (args.output_dir or _default_output_dir(run_dir, args.period, checkpoint_path)).resolve()
    summary = evaluate_saved_run(
        run_dir=run_dir,
        period=args.period,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        noah_config=args.noah_config,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
