#!/usr/bin/env python3
"""Create a 450-run extra-seed sweep from existing trained-run templates.

The sweep covers:
  - No_Negative/{center_stage5,jkge,weighted_mse}
  - lag/{lag3,lag7,lag30,no_cnn}/{center_stage5,jkge,weighted_mse}

For each scenario, 30 fresh model-initialization seeds are generated. Existing
distribution-balanced split settings are intentionally preserved from each
template config. Each generated run gets a private copy of a valid input scaler
with ``save: false`` to avoid the missing post-transfer scaler-stat issue that
has shown up in older shared scaler files.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
GRAPH_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")

CONFIG_ROOT = PROJECT_ROOT / "configs" / "extra_seed_450"
RUN_ROOT = PROJECT_ROOT / "runs" / "extra_seed_450"
SBATCH_ROOT = GRAPH_ROOT / "sbatch_extra_seed_450"
AUDIT_CSV = GRAPH_ROOT / "extra_seed_450_audit.csv"
SUBMIT_SCRIPT = GRAPH_ROOT / "Ocelote_submit_extra_seed_450.sh"

SCENARIOS: list[tuple[str, Path, Path]] = [
    ("No_Negative/center_stage5", PROJECT_ROOT / "runs/No_Negative/center_stage5", RUN_ROOT / "No_Negative/center_stage5"),
    ("No_Negative/jkge", PROJECT_ROOT / "runs/No_Negative/jkge", RUN_ROOT / "No_Negative/jkge"),
    ("No_Negative/weighted_mse", PROJECT_ROOT / "runs/No_Negative/weighted_mse", RUN_ROOT / "No_Negative/weighted_mse"),
]

for lag_name in ("lag3", "lag7", "lag30", "no_cnn"):
    for loss_name in ("center_stage5", "jkge", "weighted_mse"):
        SCENARIOS.append(
            (
                f"lag/{lag_name}/{loss_name}",
                PROJECT_ROOT / "runs/lag" / lag_name / loss_name,
                RUN_ROOT / "lag" / lag_name / loss_name,
            )
        )

VALID_SCALER_FALLBACKS = [
    PROJECT_ROOT / "scalers/noah_post_transfer_runoff_scaler_batch_ablation_weighted_mse_seed777_bs1_tgb64.yml",
]

SEEDS_PER_SCENARIO = 30
SEED_RNG = 20260509
ACCOUNTS = ["tyferre", "niug", "andrbenn", "behrangi"]
JOBS_PER_ACCOUNT = 10
MAX_MODELS_PER_JOB = 12
SBATCH_TIME = "7-00:00:00"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle, Dumper=_QuotedDigitStringDumper, sort_keys=False, default_flow_style=False)


class _QuotedDigitStringDumper(yaml.SafeDumper):
    """YAML dumper that keeps gauge IDs as strings for ruamel's safe loader."""


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "'" if value.isdigit() else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_QuotedDigitStringDumper.add_representer(str, _represent_string)


def _contains_post_transfer_stats(path: Path) -> bool:
    try:
        data = _read_yaml(path)
    except Exception:
        return False
    stats = data.get("routing_runoff_stats")
    if not isinstance(stats, dict):
        return False
    return "routing_runoff_post_transfer" in stats


def _find_template_config(template_root: Path) -> Path:
    candidates = sorted(template_root.glob("seed*/config.yml"))
    candidates += sorted(template_root.glob("seed*/saved_config.yml"))
    if not candidates:
        raise FileNotFoundError(f"No seed*/config.yml found under {template_root}")
    return candidates[0]


def _find_valid_scaler(template_config: Path, cfg: dict[str, Any]) -> Path:
    template_run = template_config.parent
    for path in sorted((template_run / "scalers").glob("*.yml")):
        if _contains_post_transfer_stats(path):
            return path

    scaler_cfg = cfg.get("scaler")
    if isinstance(scaler_cfg, dict):
        for key in ("path", "file", "scaler_file"):
            raw = scaler_cfg.get(key)
            if raw:
                path = Path(raw)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if path.exists() and _contains_post_transfer_stats(path):
                    return path

    for path in VALID_SCALER_FALLBACKS:
        if path.exists() and _contains_post_transfer_stats(path):
            return path

    raise FileNotFoundError(
        f"No valid post-transfer scaler found for {template_config}. "
        "Expected routing_runoff_stats.routing_runoff_post_transfer."
    )


def _set_seed_fields(obj: Any, seed: int) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"seed", "random_seed", "torch_seed", "numpy_seed"} and isinstance(value, int):
                obj[key] = seed
            else:
                _set_seed_fields(value, seed)
    elif isinstance(obj, list):
        for value in obj:
            _set_seed_fields(value, seed)


def _set_run_dir_fields(obj: Any, run_dir: Path) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "run_dir":
                obj[key] = str(run_dir)
            else:
                _set_run_dir_fields(value, run_dir)
    elif isinstance(obj, list):
        for value in obj:
            _set_run_dir_fields(value, run_dir)


def _increase_patience(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if "patience" in lowered and isinstance(value, int):
                obj[key] = max(value + 10, 30)
            else:
                _increase_patience(value)
    elif isinstance(obj, list):
        for value in obj:
            _increase_patience(value)


def _increase_stage_epochs(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "epochs" and isinstance(value, int):
                obj[key] = value + 40
            else:
                _increase_stage_epochs(value)
    elif isinstance(obj, list):
        for value in obj:
            _increase_stage_epochs(value)


def _configure_scaler(cfg: dict[str, Any], scaler_path: Path) -> None:
    scaler_cfg = cfg.setdefault("scaler", {})
    if not isinstance(scaler_cfg, dict):
        raise ValueError("Config has non-mapping scaler block")
    scaler_cfg["path"] = str(scaler_path)
    scaler_cfg["save"] = False


def _existing_seeds(template_root: Path) -> set[int]:
    seeds: set[int] = set()
    for path in template_root.glob("seed*"):
        suffix = path.name.removeprefix("seed")
        if suffix.isdigit():
            seeds.add(int(suffix))
    return seeds


def _make_seed_pool(scenarios: list[tuple[str, Path, Path]]) -> dict[str, list[int]]:
    rng = random.Random(SEED_RNG)
    used_global: set[int] = set()
    out: dict[str, list[int]] = {}
    for scenario_name, template_root, _ in scenarios:
        blocked = set(_existing_seeds(template_root))
        seeds: list[int] = []
        while len(seeds) < SEEDS_PER_SCENARIO:
            seed = rng.randint(100000, 999999)
            if seed in used_global or seed in blocked:
                continue
            used_global.add(seed)
            seeds.append(seed)
        out[scenario_name] = seeds
    return out


def _scenario_config_dir(scenario_name: str) -> Path:
    return CONFIG_ROOT / scenario_name


def _create_configs() -> list[dict[str, str]]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, str]] = []
    seeds_by_scenario = _make_seed_pool(SCENARIOS)

    for scenario_name, template_root, run_base in SCENARIOS:
        template_config = _find_template_config(template_root)
        template_cfg = _read_yaml(template_config)
        source_scaler = _find_valid_scaler(template_config, template_cfg)

        for seed in seeds_by_scenario[scenario_name]:
            run_dir = run_base / f"seed{seed}"
            config_path = _scenario_config_dir(scenario_name) / f"seed{seed}.yml"
            scaler_path = run_dir / "scalers" / "input_scaler.yml"

            cfg = _read_yaml(template_config)
            _set_seed_fields(cfg, seed)
            _set_run_dir_fields(cfg, run_dir)
            _increase_patience(cfg)
            _increase_stage_epochs(cfg)

            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_scaler, scaler_path)
            _configure_scaler(cfg, scaler_path)

            _write_yaml(config_path, cfg)
            run_dir.mkdir(parents=True, exist_ok=True)

            audit_rows.append(
                {
                    "scenario": scenario_name,
                    "seed": str(seed),
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "template_config": str(template_config),
                    "scaler_path": str(scaler_path),
                }
            )

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "seed", "config_path", "run_dir", "template_config", "scaler_path"],
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    return audit_rows


def _chunk_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    index = 0
    for job_index in range(len(ACCOUNTS) * JOBS_PER_ACCOUNT):
        remaining_jobs = len(ACCOUNTS) * JOBS_PER_ACCOUNT - job_index
        remaining_rows = len(rows) - index
        size = min(MAX_MODELS_PER_JOB, (remaining_rows + remaining_jobs - 1) // remaining_jobs)
        chunks.append(rows[index : index + size])
        index += size
    if index != len(rows):
        raise RuntimeError(f"Chunking mismatch: assigned {index} of {len(rows)}")
    return chunks


def _write_sbatch_scripts(rows: list[dict[str, str]]) -> None:
    SBATCH_ROOT.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_rows(rows)
    sbatches: list[Path] = []

    for job_index, chunk in enumerate(chunks, start=1):
        account = ACCOUNTS[(job_index - 1) // JOBS_PER_ACCOUNT]
        script = SBATCH_ROOT / f"Ocelote_extra_seed_450_{job_index:02d}_{account}.sh"
        sbatches.append(script)
        config_lines = "\n".join(f'  "{row["config_path"]}"' for row in chunk)
        script.write_text(
            f"""#!/bin/bash
#SBATCH --job-name=extra450_{job_index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time={SBATCH_TIME}
#SBATCH --output=/xdisk/tyferre/farmani/Graph_Routing/logs/extra450_{job_index:02d}_%j.out
#SBATCH --error=/xdisk/tyferre/farmani/Graph_Routing/logs/extra450_{job_index:02d}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=farmani@arizona.edu

set -euo pipefail

module purge
module load gnu8
module load python/3.11

PROJECT_ROOT="/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-"
cd "$PROJECT_ROOT"

mkdir -p /xdisk/tyferre/farmani/Graph_Routing/logs
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/python/3.11.4/lib:${{LD_LIBRARY_PATH:-}}
export MPLCONFIGDIR=/xdisk/tyferre/farmani/env/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

CONFIGS=(
{config_lines}
)

echo "Starting ${{#CONFIGS[@]}} sequential training runs on one GPU."
for CONFIG in "${{CONFIGS[@]}}"; do
  echo "Training config: $CONFIG"
  python -m lstm_gnn_routing.cli.main train --config-file "$CONFIG"
done
echo "Finished job {job_index:02d}."
""",
            encoding="utf-8",
        )
        script.chmod(0o755)

    SUBMIT_SCRIPT.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n\n"
        + "\n".join(f'sbatch "{path}"' for path in sbatches)
        + "\n",
        encoding="utf-8",
    )
    SUBMIT_SCRIPT.chmod(0o755)


def main() -> None:
    rows = _create_configs()
    rows = sorted(rows, key=lambda row: (row["scenario"], int(row["seed"])))
    _write_sbatch_scripts(rows)
    print(f"Created {len(rows)} configs.")
    print(f"Audit CSV: {AUDIT_CSV}")
    print(f"Config root: {CONFIG_ROOT}")
    print(f"Run root: {RUN_ROOT}")
    print(f"Sbatch root: {SBATCH_ROOT}")
    print(f"Submit helper: {SUBMIT_SCRIPT}")
    print(f"Accounts: {', '.join(ACCOUNTS)} ({JOBS_PER_ACCOUNT} jobs each; no hoshin)")


if __name__ == "__main__":
    main()
