from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lstm_gnn_routing.dataset.dataset import RoutingDataset
from lstm_gnn_routing.utils.config import RoutingConfig
from lstm_gnn_routing.utils.data import save_scaler_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute train-period forcing/static/target scalers.")
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Override scaler.path from the config")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = RoutingConfig.from_yaml(args.config_file)
    output = args.output or Path(config.section("scaler").get("path", "scalers/train_scaler.yml"))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Scaler file already exists: {output}. Pass --overwrite to replace it.")

    dataset = RoutingDataset(config, "train")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_scaler_yaml(dataset.scaler, output)
    logging.getLogger(__name__).info("Saved train scaler: %s", output)


if __name__ == "__main__":
    main()
