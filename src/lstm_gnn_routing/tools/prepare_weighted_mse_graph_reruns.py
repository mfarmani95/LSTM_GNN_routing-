from __future__ import annotations

import argparse
import copy
import csv
import shutil
from pathlib import Path
from typing import Any

import yaml
from yaml import SafeDumper


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
GRAPH_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")

EXAMPLES = {
    "graphsage": PROJECT_ROOT / "runs/graph_models/graphsage/weighted_mse/seed991116/config.yml",
    "gineconv": PROJECT_ROOT / "runs/graph_models/gineconv/weighted_mse/seed711879/config.yml",
}

SEEDS = {
    "graphsage": [184271, 239807, 318556, 427991, 503614, 618203, 704889, 812377, 936144, 982631],
    "gineconv": [173945, 265118, 349720, 451606, 590332, 637419, 728504, 840917, 915268, 997431],
}

ACCOUNTS = ["behrangi"] * 8 + ["andrbenn"] * 8 + ["niug"] * 4
VALID_SCALER = (
    PROJECT_ROOT
    / "scalers/noah_post_transfer_runoff_scaler_batch_ablation_weighted_mse_seed777_bs1_tgb64.yml"
)


def _recursive_set_seed(obj: Any, seed: int, path: tuple[str, ...] = ()) -> Any:
    """Set model/training RNG seeds without changing data-split seeds.

    The reruns should use the exact same train/validation/test split as the
    template runs.  Many configs use keys such as split_seed or data_split.seed
    to point to a fixed split.  Those must remain untouched.  We only update
    generic model/training RNG seed keys outside split/scaler/sample contexts.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            low = str(key).lower()
            key_path = path + (low,)
            protected = any(
                token in part
                for part in key_path
                for token in ("split", "block", "sample", "scaler", "lookup")
            )
            if not protected and low in {"seed", "random_seed", "torch_seed", "numpy_seed", "model_seed"}:
                out[key] = int(seed)
            else:
                out[key] = _recursive_set_seed(value, seed, key_path)
        return out
    if isinstance(obj, list):
        return [_recursive_set_seed(value, seed, path) for value in obj]
    return obj


def _replace_paths(obj: Any, old_run_dir: str, new_run_dir: str, old_seed: str, new_seed: str) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_paths(value, old_run_dir, new_run_dir, old_seed, new_seed) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_paths(value, old_run_dir, new_run_dir, old_seed, new_seed) for value in obj]
    if isinstance(obj, str):
        text = obj.replace(old_run_dir, new_run_dir)
        # Keep split/scaler file names untouched if they encode split seed; only
        # replace paths that still point at the old run directory.
        if old_run_dir in obj:
            text = text.replace(old_seed, new_seed)
        return text
    return obj


def _set_run_dir(cfg: dict[str, Any], run_dir: Path) -> None:
    for key in ("run_dir", "output_dir", "save_dir", "experiment_dir"):
        if key in cfg:
            cfg[key] = str(run_dir)
    if "training" in cfg and isinstance(cfg["training"], dict):
        for key in ("run_dir", "output_dir", "save_dir", "experiment_dir"):
            if key in cfg["training"]:
                cfg["training"][key] = str(run_dir)
    # Most configs in this repo use a top-level run_dir. Add it if absent so
    # reruns are organized even when the template relied on defaults.
    cfg.setdefault("run_dir", str(run_dir))


def _set_private_scaler(cfg: dict[str, Any], run_dir: Path) -> Path:
    scaler_dir = run_dir / "scalers"
    scaler_path = scaler_dir / "input_scaler.yml"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VALID_SCALER, scaler_path)
    cfg["scaler"] = {
        "path": str(scaler_path),
        "save": False,
    }
    return scaler_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return cfg


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    class _QuotedNumericStringDumper(SafeDumper):
        pass

    def _represent_str(dumper: yaml.Dumper, data: str) -> yaml.Node:
        style = "'" if data.strip().isdigit() else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _QuotedNumericStringDumper.add_representer(str, _represent_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(cfg, handle, Dumper=_QuotedNumericStringDumper, sort_keys=False, width=120)


def _flatten(obj: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out.update(_flatten(value, prefix + (str(key),)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, value in enumerate(obj):
            out.update(_flatten(value, prefix + (str(i),)))
        return out
    return {".".join(prefix): obj}


def _split_like_entries(cfg: dict[str, Any]) -> dict[str, str]:
    flat = _flatten(cfg)
    entries = {}
    for key, value in flat.items():
        low = key.lower()
        text = str(value)
        if any(token in low or token in text.lower() for token in ("split", "block", "sample_lookup")):
            entries[key] = text
    return entries


def _seed_entries(cfg: dict[str, Any]) -> dict[str, str]:
    flat = _flatten(cfg)
    return {key: str(value) for key, value in flat.items() if key.lower().split(".")[-1] in {"seed", "random_seed", "torch_seed", "numpy_seed", "model_seed"}}


def _sbatch_text(account: str, model: str, seed: int, config_path: Path) -> str:
    name = f"{model}_wmse_{seed}"
    log_dir = GRAPH_ROOT / "logs/graph_weighted_mse_reruns"
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output={log_dir}/{name}_%j.out
#SBATCH --error={log_dir}/{name}_%j.err
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
    parser = argparse.ArgumentParser(description="Prepare weighted-MSE GraphSAGE/GINEConv reruns from working configs.")
    parser.add_argument("--config-root", type=Path, default=PROJECT_ROOT / "configs/graph_weighted_mse_reruns")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs/graph_weighted_mse_reruns")
    parser.add_argument("--sbatch-root", type=Path, default=GRAPH_ROOT / "sbatch_graph_weighted_mse_reruns")
    args = parser.parse_args()

    args.sbatch_root.mkdir(parents=True, exist_ok=True)
    submit_paths: list[Path] = []
    audit_rows: list[dict[str, str]] = []
    job_index = 0
    for model, example_path in EXAMPLES.items():
        template = _load_yaml(example_path)
        template_split_entries = _split_like_entries(template)
        template_seed_entries = _seed_entries(template)
        old_seed = example_path.parent.name.replace("seed", "")
        old_run_dir = str(example_path.parent)
        for seed in SEEDS[model]:
            run_dir = args.run_root / model / "weighted_mse" / f"seed{seed}"
            config_path = args.config_root / model / "weighted_mse" / f"seed{seed}.yml"
            cfg = copy.deepcopy(template)
            cfg = _recursive_set_seed(cfg, seed)
            cfg = _replace_paths(cfg, old_run_dir, str(run_dir), old_seed, str(seed))
            _set_run_dir(cfg, run_dir)
            scaler_path = _set_private_scaler(cfg, run_dir)
            _write_yaml(config_path, cfg)

            new_split_entries = _split_like_entries(cfg)
            new_seed_entries = _seed_entries(cfg)
            split_match = template_split_entries == new_split_entries
            audit_rows.append(
                {
                    "model": model,
                    "seed": str(seed),
                    "template": str(example_path),
                    "config": str(config_path),
                    "run_dir": str(run_dir),
                    "scaler_path": str(scaler_path),
                    "split_entries_match_template": str(split_match),
                    "template_seed_entries": repr(template_seed_entries),
                    "new_seed_entries": repr(new_seed_entries),
                    "template_split_entries": repr(template_split_entries),
                    "new_split_entries": repr(new_split_entries),
                }
            )

            account = ACCOUNTS[job_index % len(ACCOUNTS)]
            sbatch_path = args.sbatch_root / f"Ocelote_train_{model}_weighted_mse_seed{seed}.sh"
            sbatch_path.write_text(_sbatch_text(account, model, seed, config_path), encoding="utf-8")
            submit_paths.append(sbatch_path)
            job_index += 1

    submit_script = GRAPH_ROOT / "Ocelote_submit_graph_weighted_mse_reruns.sh"
    submit_script.write_text(
        "#!/bin/bash\nset -euo pipefail\n\n"
        + "\n".join(f"sbatch {path}" for path in submit_paths)
        + "\n",
        encoding="utf-8",
    )
    audit_path = GRAPH_ROOT / "graph_weighted_mse_reruns_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote {len(submit_paths)} sbatch files to {args.sbatch_root}")
    print(f"Wrote submit helper: {submit_script}")
    print(f"Wrote audit report: {audit_path}")


if __name__ == "__main__":
    main()
