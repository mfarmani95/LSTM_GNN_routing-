from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lstm_gnn_routing.training.evaluate import main as evaluate_main
from lstm_gnn_routing.training.train import start_training
from lstm_gnn_routing.utils.config import RoutingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train standalone LSTM runoff + GNN routing models.")
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config-file", required=True, type=Path, help="Path to YAML config")
    train_parser.add_argument("--noah-config", type=str, default=None, help="Optional Noah config for noah_table_priors static features")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved run on validation/test data")
    evaluate_parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing config.yml and checkpoints")
    evaluate_parser.add_argument("--period", default="test", choices=("validation", "test"), help="Dataset split to evaluate")
    evaluate_parser.add_argument(
        "--checkpoint-file",
        default="best_final_stage_model.pt",
        help="Checkpoint filename relative to run-dir, or an absolute checkpoint path",
    )
    evaluate_parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for evaluation outputs")
    evaluate_parser.add_argument("--noah-config", type=str, default=None, help="Optional Noah config for noah_table_priors static features")
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
        if args.noah_config is not None:
            config.set_noah_config_path(args.noah_config)
        return start_training(config)
    if args.command == "evaluate":
        evaluate_argv = [
            "--run-dir",
            str(args.run_dir),
            "--period",
            str(args.period),
            "--checkpoint-file",
            str(args.checkpoint_file),
        ]
        if args.output_dir is not None:
            evaluate_argv.extend(["--output-dir", str(args.output_dir)])
        if args.noah_config is not None:
            evaluate_argv.extend(["--noah-config", str(args.noah_config)])
        return evaluate_main(evaluate_argv)
    parser.print_help()
    return None


if __name__ == "__main__":
    main()
