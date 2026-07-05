from __future__ import annotations

import argparse
import csv
import copy
import shutil
from pathlib import Path
from typing import Any

import yaml
from yaml import SafeDumper


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
GRAPH_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
TEMPLATE_CONFIG = PROJECT_ROOT / "runs/No_Negative/jkge/seed286876/config.yml"
VALID_SCALER = (
    PROJECT_ROOT
    / "scalers/noah_post_transfer_runoff_scaler_batch_ablation_weighted_mse_seed777_bs1_tgb64.yml"
)
SEEDS = [116203, 207941, 334572, 486019, 529884, 671250, 734908, 816337, 902146, 975531]
ACCOUNTS = ["behrangi", "behrangi", "behrangi", "andrbenn", "andrbenn", "andrbenn", "niug", "niug", "hoshin", "hoshin"]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return cfg


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    class Dumper(SafeDumper):
        pass

    def represent_str(dumper: yaml.Dumper, value: str) -> yaml.Node:
        style = "'" if value.strip().isdigit() else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    Dumper.add_representer(str, represent_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(cfg, handle, Dumper=Dumper, sort_keys=False, width=120)


def _flatten(obj: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out.update(_flatten(value, prefix + (str(key),)))
        return out
    if isinstance(obj, list):
        out = {}
        for idx, value in enumerate(obj):
            out.update(_flatten(value, prefix + (str(idx),)))
        return out
    return {".".join(prefix): obj}


def _split_like_entries(cfg: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key, value in _flatten(cfg).items():
        low_key = key.lower()
        if any(token in low_key for token in ("split_file", "split_path", "split_seed", "data_split", "block", "sample_lookup")):
            out[key] = str(value)
    return out


def _set_private_scaler(cfg: dict[str, Any], run_dir: Path) -> Path:
    scaler_path = run_dir / "scalers" / "input_scaler.yml"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VALID_SCALER, scaler_path)
    cfg["scaler"] = {"path": str(scaler_path), "save": False}
    return scaler_path


def _sbatch_text(config_path: Path, account: str, seed: int) -> str:
    log_dir = GRAPH_ROOT / "logs/no_negative_jkge_extra_seeds"
    return f"""#!/bin/bash
#SBATCH --job-name=nn_jkge_{seed}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output={log_dir}/nn_jkge_{seed}_%j.out
#SBATCH --error={log_dir}/nn_jkge_{seed}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=farmani@arizona.edu

set -euo pipefail

module purge
module load gnu8
module load python/3.11

cd {PROJECT_ROOT}
mkdir -p {log_dir}
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/python/3.11.4/lib:${{LD_LIBRARY_PATH:-}}
export MPLCONFIGDIR=/xdisk/tyferre/farmani/env/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

python -m lstm_gnn_routing.cli.main train --config-file {config_path}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 10 extra No_Negative/jkge seed runs.")
    parser.add_argument("--config-root", type=Path, default=PROJECT_ROOT / "configs/no_negative_jkge_extra_seeds")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs/No_Negative/jkge")
    parser.add_argument("--sbatch-root", type=Path, default=GRAPH_ROOT / "sbatch_no_negative_jkge_extra_seeds")
    args = parser.parse_args()

    template = _load_yaml(TEMPLATE_CONFIG)
    template_split = _split_like_entries(template)
    args.sbatch_root.mkdir(parents=True, exist_ok=True)

    submit_paths: list[Path] = []
    audit_rows: list[dict[str, str]] = []
    for seed, account in zip(SEEDS, ACCOUNTS):
        cfg = copy.deepcopy(template)
        run_dir = args.run_root / f"seed{seed}"
        config_path = args.config_root / f"seed{seed}.yml"

        cfg["experiment_name"] = f"no_negative_jkge_split619873_seed{seed}"
        cfg["run_dir"] = str(run_dir)
        cfg.setdefault("training", {})["seed"] = int(seed)
        scaler_path = _set_private_scaler(cfg, run_dir)

        split_match = _split_like_entries(cfg) == template_split
        _write_yaml(config_path, cfg)

        sbatch_path = args.sbatch_root / f"Ocelote_train_no_negative_jkge_seed{seed}.sh"
        sbatch_path.write_text(_sbatch_text(config_path, account, seed), encoding="utf-8")
        submit_paths.append(sbatch_path)
        audit_rows.append(
            {
                "seed": str(seed),
                "account": account,
                "config": str(config_path),
                "run_dir": str(run_dir),
                "scaler_path": str(scaler_path),
                "scaler_save": "False",
                "split_entries_match_template": str(split_match),
            }
        )

    submit_script = GRAPH_ROOT / "Ocelote_submit_no_negative_jkge_extra_seeds.sh"
    submit_script.write_text(
        "#!/bin/bash\nset -euo pipefail\n\n"
        + "\n".join(f"sbatch {path}" for path in submit_paths)
        + "\n",
        encoding="utf-8",
    )

    audit_path = GRAPH_ROOT / "no_negative_jkge_extra_seeds_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Wrote {len(submit_paths)} sbatch files to {args.sbatch_root}")
    print(f"Wrote submit helper: {submit_script}")
    print(f"Wrote audit report: {audit_path}")


if __name__ == "__main__":
    main()
