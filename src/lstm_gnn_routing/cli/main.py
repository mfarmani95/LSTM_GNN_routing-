from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lstm_gnn_routing.training.train import start_training
from lstm_gnn_routing.utils.config import RoutingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train standalone LSTM runoff + GNN routing models.")
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config-file", required=True, type=Path, help="Path to YAML config")
    return parser


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        config = RoutingConfig.from_yaml(args.config_file)
        return start_training(config)
    parser.print_help()
    return None


if __name__ == "__main__":
    main()
