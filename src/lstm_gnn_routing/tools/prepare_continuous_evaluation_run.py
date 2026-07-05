from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ruamel.yaml import YAML


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight run directory for continuous inference. "
            "The copied config uses period='test' as the full continuous date range, "
            "disables precomputed block splitting, and preserves the trained model settings."
        )
    )
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--output-run-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sequence-days", type=int, default=None)
    parser.add_argument("--stride-days", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--checkpoint-files",
        nargs="*",
        default=["best_final_stage_model.pt", "best_model.pt", "last_model.pt"],
        help="Checkpoint files to symlink/copy from the source run when present.",
    )
    parser.add_argument("--copy-checkpoints", action="store_true", help="Copy checkpoints instead of symlinking.")
    return parser.parse_args()


def _copy_or_link(src: Path, dst: Path, *, copy: bool) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> None:
    args = _parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()
    source_config = source_run_dir / "config.yml"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing source config: {source_config}")

    yaml = YAML()
    yaml.preserve_quotes = True
    with source_config.open() as fp:
        config = yaml.load(fp)

    output_run_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_name"] = args.experiment_name or f"{config.get('experiment_name', source_run_dir.name)}_continuous"
    config["run_dir"] = str(output_run_dir)

    # Reuse the dataset's test period machinery, but make the test period the
    # continuous full-range period and disable block-split filtering.
    config["test_start_date"] = args.start_date
    config["test_end_date"] = args.end_date
    config["data_split"] = {"type": "chronological"}
    config.setdefault("windowing", {})
    config["windowing"]["apply_to_validation_test"] = True
    if args.sequence_days is not None:
        config["windowing"]["sequence_days"] = int(args.sequence_days)
    if args.stride_days is not None:
        config["windowing"]["stride_days"] = int(args.stride_days)

    with (output_run_dir / "config.yml").open("w") as fp:
        yaml.dump(config, fp)

    for checkpoint_file in args.checkpoint_files:
        _copy_or_link(source_run_dir / checkpoint_file, output_run_dir / checkpoint_file, copy=args.copy_checkpoints)

    print(f"Prepared continuous evaluation run: {output_run_dir}")
    print(f"Config: {output_run_dir / 'config.yml'}")


if __name__ == "__main__":
    main()
