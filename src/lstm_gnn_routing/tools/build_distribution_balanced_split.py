"""Build reusable distribution-balanced 180-day streamflow block splits.

The split is intentionally block-based: each non-overlapping block is assigned to
train, validation, or test once, and the dataset can later create overlapping
windows only inside same-period contiguous block segments.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from lstm_gnn_routing.utils.config import RoutingConfig
from lstm_gnn_routing.utils.data import load_basin_file, load_csv_targets

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path, help="Routing YAML config to read target settings from.")
    parser.add_argument("--basin-file", type=Path, default=None, help="Optional basin list; defaults to config train_basin_file.")
    parser.add_argument("--output", required=True, type=Path, help="Output YAML split file.")
    parser.add_argument("--report-csv", type=Path, default=None, help="Optional per-block CSV report path.")
    parser.add_argument("--block-days", type=int, default=180, help="Non-overlapping block length in days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie-breaking during greedy assignment.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-valid-fraction", type=float, default=0.50, help="Minimum finite target fraction required per block.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional global split start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional global split end date.")
    parser.add_argument(
        "--context-days",
        type=int,
        default=None,
        help="Prediction context days to reserve before first block; defaults to max routing/runoff lag from config.",
    )
    return parser.parse_args()


def _default_context_days(config: RoutingConfig) -> int:
    routing_model = config.section("routing_model")
    runoff_model = config.section("runoff_model")
    routing_context = int(routing_model.get("routing_lag_context_days", 0) or 0)
    runoff_warmup = int(runoff_model.get("warmup_days", runoff_model.get("lstm_warmup_days", 0)) or 0)
    return max(routing_context, runoff_warmup)


def _load_targets_for_basins(config: RoutingConfig, basin_ids: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frames: list[pd.DataFrame] = []
    for basin_id in basin_ids:
        path = Path(config.target_dir) / config.target_file_pattern.format(basin_id=basin_id)
        frames.append(
            load_csv_targets(
                file_path=path,
                date_column=config.target_date_column,
                target_variables=config.target_variables,
                separator=config.target_separator,
                basin_id_column=config.target_basin_id_column,
                unit_conversion=config.target_unit_conversion,
            )
        )

    combined_index = frames[0].index
    for frame in frames[1:]:
        combined_index = combined_index.union(frame.index)
    combined_index = pd.DatetimeIndex(combined_index.sort_values())

    arrays = [
        frame.reindex(combined_index)[config.target_variables].to_numpy(dtype=np.float64, copy=False)
        for frame in frames
    ]
    targets = np.stack(arrays, axis=1)
    return combined_index, targets


def _safe_nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, percentile))


def _block_features(block_values: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    values = block_values[..., 0] if block_values.ndim == 3 else block_values
    finite = np.isfinite(values)
    valid_fraction = float(finite.mean()) if finite.size else 0.0
    log_values = np.log1p(np.clip(values.astype(np.float64), a_min=0.0, a_max=None))
    aggregate_daily = np.nanmean(log_values, axis=1)
    gauge_means = np.nanmean(log_values, axis=0)
    gauge_stds = np.nanstd(log_values, axis=0)
    midpoint = start + (end - start) / 2
    day_angle = 2.0 * np.pi * (float(midpoint.dayofyear) - 1.0) / 366.0
    return {
        "valid_fraction": valid_fraction,
        "mean_log_flow": float(np.nanmean(log_values)),
        "p10_log_flow": _safe_nanpercentile(log_values, 10.0),
        "p90_log_flow": _safe_nanpercentile(log_values, 90.0),
        "daily_std_log_flow": float(np.nanstd(aggregate_daily)),
        "gauge_mean_spread": float(np.nanstd(gauge_means)),
        "gauge_std_mean": float(np.nanmean(gauge_stds)),
        "season_sin": float(np.sin(day_angle)),
        "season_cos": float(np.cos(day_angle)),
    }


def _target_counts(n_blocks: int, fractions: dict[str, float]) -> dict[str, int]:
    names = ["train", "validation", "test"]
    raw = {name: float(fractions[name]) * n_blocks for name in names}
    counts = {name: int(np.floor(raw[name])) for name in names}
    remaining = n_blocks - sum(counts.values())
    for name in sorted(names, key=lambda item: raw[item] - counts[item], reverse=True):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1
    return counts


def _scaled_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    p25 = np.nanpercentile(x, 25.0, axis=0)
    p75 = np.nanpercentile(x, 75.0, axis=0)
    scale = np.where(np.abs(p75 - p25) > 1.0e-8, p75 - p25, np.nanstd(x, axis=0))
    scale = np.where(np.abs(scale) > 1.0e-8, scale, 1.0)
    return (x - med) / scale


def _assign_blocks(frame: pd.DataFrame, seed: int, fractions: dict[str, float]) -> list[str]:
    periods = ["train", "validation", "test"]
    feature_columns = [
        "mean_log_flow",
        "p10_log_flow",
        "p90_log_flow",
        "daily_std_log_flow",
        "gauge_mean_spread",
        "gauge_std_mean",
        "season_sin",
        "season_cos",
    ]
    x = _scaled_feature_matrix(frame, feature_columns)
    global_mean = np.nanmean(x, axis=0)
    global_std = np.nanstd(x, axis=0)
    target_counts = _target_counts(len(frame), fractions)
    rng = np.random.default_rng(seed)

    labels = np.asarray(
        [period for period in periods for _ in range(target_counts[period])],
        dtype=object,
    )
    if len(labels) != len(frame):
        raise RuntimeError("Internal split assignment error: label count does not match block count.")

    def score_assignment(assignment_array: np.ndarray) -> float:
        score = 0.0
        for period in periods:
            subset = x[assignment_array == period]
            if subset.size == 0:
                return float("inf")
            mean_score = np.linalg.norm(np.nanmean(subset, axis=0) - global_mean)
            std_score = np.linalg.norm(np.nanstd(subset, axis=0) - global_std)
            score += float(mean_score + 0.25 * std_score)
        return score

    best_assignment: np.ndarray | None = None
    best_score = float("inf")
    n_random_starts = 2000
    for _ in range(n_random_starts):
        candidate = labels.copy()
        rng.shuffle(candidate)
        candidate_score = score_assignment(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_assignment = candidate.copy()

    if best_assignment is None:
        raise RuntimeError("Could not create a balanced split assignment.")

    # Refine by swapping labels between two blocks when the swap improves balance.
    improved = True
    max_passes = 25
    pass_count = 0
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        order = rng.permutation(len(best_assignment))
        for left_pos, left in enumerate(order):
            for right in order[left_pos + 1 :]:
                if best_assignment[left] == best_assignment[right]:
                    continue
                candidate = best_assignment.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = score_assignment(candidate)
                if candidate_score + 1.0e-10 < best_score:
                    best_score = candidate_score
                    best_assignment = candidate
                    improved = True
                    break
            if improved:
                break

    return [str(value) for value in best_assignment.tolist()]


def _period_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    feature_columns = [
        "mean_log_flow",
        "p10_log_flow",
        "p90_log_flow",
        "daily_std_log_flow",
        "gauge_mean_spread",
        "gauge_std_mean",
        "season_sin",
        "season_cos",
    ]
    for period, group in frame.groupby("period", sort=True):
        summary[str(period)] = {
            "blocks": int(len(group)),
            "start": str(group["start"].min()),
            "end": str(group["end"].max()),
            "features": {
                column: float(group[column].mean())
                for column in feature_columns
            },
        }
    return summary


def build_split(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    config = RoutingConfig.from_yaml(args.config_file)
    basin_file = args.basin_file or config.train_basin_file
    basin_ids = [str(value) for value in load_basin_file(basin_file)]
    time_index, targets = _load_targets_for_basins(config, basin_ids)

    global_start = pd.Timestamp(args.start_date) if args.start_date else min(
        config.train_start_date,
        config.validation_start_date,
        config.test_start_date,
    )
    global_end = pd.Timestamp(args.end_date) if args.end_date else max(
        config.train_end_date,
        config.validation_end_date,
        config.test_end_date,
    )
    context_days = _default_context_days(config) if args.context_days is None else int(args.context_days)
    first_block_start = global_start + pd.DateOffset(days=context_days)
    final_end = global_end + pd.Timedelta(days=1)

    records: list[dict[str, Any]] = []
    block_start = first_block_start
    block_id = 0
    while block_start + pd.DateOffset(days=args.block_days) <= final_end:
        block_end = block_start + pd.DateOffset(days=args.block_days)
        mask = (time_index >= block_start) & (time_index < block_end)
        if int(mask.sum()) > 0:
            features = _block_features(targets[mask], block_start, block_end)
            if features["valid_fraction"] >= float(args.min_valid_fraction):
                records.append(
                    {
                        "block_id": block_id,
                        "start": block_start.strftime("%Y-%m-%d"),
                        "end": block_end.strftime("%Y-%m-%d"),
                        "n_days": int(mask.sum()),
                        **features,
                    }
                )
        block_id += 1
        block_start = block_end

    if len(records) < 3:
        raise RuntimeError("Not enough valid blocks were created to assign train/validation/test splits.")

    frame = pd.DataFrame.from_records(records)
    fractions = {
        "train": float(args.train_fraction),
        "validation": float(args.validation_fraction),
        "test": float(args.test_fraction),
    }
    total_fraction = sum(fractions.values())
    if not np.isclose(total_fraction, 1.0):
        fractions = {key: value / total_fraction for key, value in fractions.items()}
    frame["period"] = _assign_blocks(frame, args.seed, fractions)

    blocks = []
    for row in frame.sort_values("block_id").to_dict(orient="records"):
        blocks.append(
            {
                "block_id": int(row["block_id"]),
                "period": str(row["period"]),
                "start": str(row["start"]),
                "end": str(row["end"]),
                "n_days": int(row["n_days"]),
                "valid_fraction": float(row["valid_fraction"]),
                "features": {
                    key: float(row[key])
                    for key in [
                        "mean_log_flow",
                        "p10_log_flow",
                        "p90_log_flow",
                        "daily_std_log_flow",
                        "gauge_mean_spread",
                        "gauge_std_mean",
                        "season_sin",
                        "season_cos",
                    ]
                },
            }
        )

    payload = {
        "metadata": {
            "split_type": "distribution_balanced_nonoverlapping_blocks",
            "block_days": int(args.block_days),
            "seed": int(args.seed),
            "fractions": fractions,
            "context_days": int(context_days),
            "global_start": global_start.strftime("%Y-%m-%d"),
            "global_end": global_end.strftime("%Y-%m-%d"),
            "first_block_start": first_block_start.strftime("%Y-%m-%d"),
            "basin_file": str(basin_file),
            "basin_ids": basin_ids,
            "target_variables": list(config.target_variables),
            "min_valid_fraction": float(args.min_valid_fraction),
            "assignment_summary": _period_summary(frame),
        },
        "blocks": blocks,
    }
    return payload, frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    payload, frame = build_split(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    with args.output.open("w") as fp:
        yaml.dump(payload, fp)

    report_csv = args.report_csv or args.output.with_suffix(".csv")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_csv, index=False)

    logger.info("Wrote split YAML: %s", args.output)
    logger.info("Wrote split report CSV: %s", report_csv)
    logger.info("Assignment summary: %s", json.dumps(payload["metadata"]["assignment_summary"], indent=2))


if __name__ == "__main__":
    main()
