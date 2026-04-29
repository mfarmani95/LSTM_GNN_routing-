from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lstm_gnn_routing.dataset.batcher import RoutingBatcher
from lstm_gnn_routing.dataset.dataset import RoutingDataset
from lstm_gnn_routing.training.model_factory import build_runoff_model, build_runoff_transfer_model
from lstm_gnn_routing.utils.config import RoutingConfig
from lstm_gnn_routing.utils.data import save_scaler_yaml


def _trim_runoff_outputs_for_routing(
    runoff_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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

    trimmed: dict[str, torch.Tensor] = {}
    for key, value in runoff_outputs.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            trimmed[key] = value
            continue
        if value.shape[1] <= first_trim:
            raise ValueError(
                f"Runoff warm-up trim length {first_trim} is not smaller than runoff output time length "
                f"{value.shape[1]} for '{key}'."
            )
        trimmed[key] = value[:, first_trim:]
    return trimmed


def _apply_runoff_transform(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    transform = str(transform or "identity").lower()
    if transform in {"", "none", "identity"}:
        return tensor
    if transform == "log1p":
        return torch.log1p(torch.clamp(tensor, min=0.0))
    raise ValueError("routing_model.runoff_input_transform must be one of: identity, log1p")


def _compute_routing_runoff_stats(config: RoutingConfig, dataset: RoutingDataset) -> None:
    routing_cfg = config.section("routing_model")
    if not bool(routing_cfg.get("normalize_runoff_inputs", False)):
        return

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=RoutingBatcher.collate_fn,
        pin_memory=False,
    )
    example_batch = next(iter(loader))
    device = torch.device("cpu")
    runoff_model = build_runoff_model(config, example_batch=example_batch, device=device)
    runoff_transfer = build_runoff_transfer_model(config, example_batch=example_batch, device=device)
    runoff_model.eval()
    if runoff_transfer is not None:
        runoff_transfer.eval()

    selected_keys = tuple(str(key) for key in routing_cfg.get("runoff_output_keys", ["runoff_total"]))
    transform = str(routing_cfg.get("runoff_input_transform", "identity")).lower()
    sums: dict[str, float] = {key: 0.0 for key in selected_keys}
    sums_sq: dict[str, float] = {key: 0.0 for key in selected_keys}
    counts: dict[str, int] = {key: 0 for key in selected_keys}

    logger = logging.getLogger(__name__)
    logger.info("Computing routing runoff stats for keys=%s", selected_keys)
    with torch.no_grad():
        for batch in loader:
            runoff_outputs = runoff_model(batch)
            trimmed_outputs = _trim_runoff_outputs_for_routing(runoff_outputs, batch)
            routed_outputs = (
                runoff_transfer(trimmed_outputs, {"routing_graph": batch["routing_graph"]})
                if runoff_transfer is not None
                else trimmed_outputs
            )
            for key in selected_keys:
                if key not in routed_outputs:
                    raise KeyError(f"Routing runoff stats requested key '{key}' but it was not produced by the pipeline.")
                tensor = routed_outputs[key]
                if not isinstance(tensor, torch.Tensor):
                    continue
                values = _apply_runoff_transform(tensor.detach().to(dtype=torch.float32, device=device), transform)
                finite = torch.isfinite(values)
                if not torch.any(finite):
                    continue
                selected = values[finite].to(dtype=torch.float64)
                sums[key] += float(selected.sum().item())
                sums_sq[key] += float(torch.square(selected).sum().item())
                counts[key] += int(selected.numel())

    group_name = str(routing_cfg.get("runoff_scaler_group", "routing_runoff"))
    keys_stats: dict[str, dict[str, float | str]] = {}
    for key in selected_keys:
        if counts[key] <= 0:
            raise RuntimeError(f"No finite transferred runoff values were found for key '{key}'.")
        mean = sums[key] / float(counts[key])
        variance = max(sums_sq[key] / float(counts[key]) - mean * mean, 0.0)
        std = max(variance ** 0.5, 1.0e-6)
        keys_stats[key] = {
            "mean": float(mean),
            "std": float(std),
            "transform": transform,
        }

    dataset.scaler.setdefault("routing_runoff_stats", {})[group_name] = {
        "transform": transform,
        "keys": keys_stats,
    }
    logger.info("Computed routing runoff stats group '%s': %s", group_name, keys_stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute train-period forcing/static/target scalers.")
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Override scaler.path from the config")
    parser.add_argument("--noah-config", type=str, default=None, help="Optional Noah config for noah_table_priors static features")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = RoutingConfig.from_yaml(args.config_file)
    if args.noah_config is not None:
        config.set_noah_config_path(args.noah_config)
    output = args.output or Path(config.section("scaler").get("path", "scalers/train_scaler.yml"))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Scaler file already exists: {output}. Pass --overwrite to replace it.")

    dataset = RoutingDataset(config, "train")
    _compute_routing_runoff_stats(config, dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_scaler_yaml(dataset.scaler, output)
    logging.getLogger(__name__).info("Saved train scaler: %s", output)


if __name__ == "__main__":
    main()
