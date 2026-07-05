"""Prepare one lag/CNN sensitivity training config.

This helper starts from an existing candidate config, copies the split-related
settings from a reference best run, then writes a scenario-specific config.
It keeps the model/loss/graph choices from the candidate while ensuring every
candidate uses the same train/validation/test split for a fair comparison.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


class _QuotedNumericStringDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    # ruamel.yaml interprets unquoted leading-zero gauge IDs as integers.
    # Quote digit-only strings so gauge IDs stay as strings when configs load.
    style = '"' if data.isdigit() else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_QuotedNumericStringDumper.add_representer(str, _represent_string)


SPLIT_TOKENS = (
    "split",
    "block_assignment",
    "balanced_block",
    "distribution_balanced",
)

LOSS_TOKENS = (
    "loss",
    "criterion",
)

SUPERVISION_KEYS = (
    "targets",
    "target",
    "curriculum",
    "stages",
    "curriculum_stages",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _copy_split_settings(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively copy split-related settings from src into dst."""

    for key, src_value in src.items():
        lowered = str(key).lower()
        if any(token in lowered for token in SPLIT_TOKENS):
            dst[key] = copy.deepcopy(src_value)
            continue

        dst_value = dst.get(key)
        if isinstance(src_value, dict) and isinstance(dst_value, dict):
            _copy_split_settings(dst_value, src_value)


def _copy_loss_settings(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively copy loss-related settings from src into dst."""

    for key, src_value in src.items():
        lowered = str(key).lower()
        if any(token in lowered for token in LOSS_TOKENS):
            dst[key] = copy.deepcopy(src_value)
            continue

        dst_value = dst.get(key)
        if isinstance(src_value, dict) and isinstance(dst_value, dict):
            _copy_loss_settings(dst_value, src_value)


def _copy_supervision_settings(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Copy target/curriculum settings that must match the dataset gauges."""

    for key, src_value in src.items():
        lowered = str(key).lower()
        if lowered in SUPERVISION_KEYS:
            dst[key] = copy.deepcopy(src_value)
            continue

        dst_value = dst.get(key)
        if isinstance(src_value, dict) and isinstance(dst_value, dict):
            _copy_supervision_settings(dst_value, src_value)


def _set_seed(cfg: dict[str, Any], seed: int) -> None:
    cfg["seed"] = seed
    cfg["random_seed"] = seed
    training = cfg.setdefault("training", {})
    if isinstance(training, dict):
        training["seed"] = seed


def _set_routing_options(
    cfg: dict[str, Any],
    runoff_lags: list[int],
    routing_lag_context_days: int,
    temporal_head: str,
) -> None:
    routing_model = cfg.setdefault("routing_model", {})
    if not isinstance(routing_model, dict):
        raise ValueError("routing_model must be a mapping")

    routing_model["runoff_lags"] = runoff_lags
    routing_model["routing_lag_context_days"] = routing_lag_context_days
    routing_model["temporal_head"] = temporal_head


def _parse_lags(raw: str) -> list[int]:
    lags = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not lags:
        raise ValueError("At least one runoff lag is required")
    return lags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--reference-split-config", required=True, type=Path)
    parser.add_argument("--reference-loss-config", type=Path)
    parser.add_argument("--reference-supervision-config", type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--runoff-lags", required=True)
    parser.add_argument("--routing-lag-context-days", required=True, type=int)
    parser.add_argument("--temporal-head", required=True)
    args = parser.parse_args()

    cfg = _stringify_mapping_keys(_load_yaml(args.source_config))
    split_cfg = _load_yaml(args.reference_split_config)

    _copy_split_settings(cfg, split_cfg)
    if args.reference_loss_config is not None:
        loss_cfg = _load_yaml(args.reference_loss_config)
        _copy_loss_settings(cfg, loss_cfg)
    if args.reference_supervision_config is not None:
        supervision_cfg = _load_yaml(args.reference_supervision_config)
        _copy_supervision_settings(cfg, supervision_cfg)
    _set_seed(cfg, args.seed)
    _set_routing_options(
        cfg,
        runoff_lags=_parse_lags(args.runoff_lags),
        routing_lag_context_days=args.routing_lag_context_days,
        temporal_head=args.temporal_head,
    )

    cfg["experiment_name"] = args.experiment_name
    cfg["run_dir"] = str(args.run_dir)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as fp:
        yaml.dump(
            cfg,
            fp,
            Dumper=_QuotedNumericStringDumper,
            sort_keys=False,
            default_flow_style=False,
        )


if __name__ == "__main__":
    main()
