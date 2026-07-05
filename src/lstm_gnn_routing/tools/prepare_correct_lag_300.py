#!/usr/bin/env python3
"""Create dense-lag rerun configs from the existing extra_seed_450 templates.

The previous lag experiments used sparse lag sets, e.g. [0, 1, 2, 3, 7, 14].
This generator creates the corrected dense lag experiments where all daily lags
from 0 through the requested maximum lag are included.

Scenarios:
  - No_Negative/lag14/{center_stage5,jkge,weighted_mse}
  - lag/lag7/{center_stage5,jkge,weighted_mse}
  - lag/lag30/{center_stage5,jkge,weighted_mse}
  - lag/no_cnn_lag14/{center_stage5,jkge,weighted_mse}

Each scenario/loss pair gets 20 new random seeds. Existing data split settings,
loss settings, graph settings, and model choices are preserved from the template
configs. Each run gets a private copy of a valid scaler with save=false.
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

CONFIG_ROOT = PROJECT_ROOT / "configs" / "correct_lag"
RUN_ROOT = PROJECT_ROOT / "runs" / "correct_lag"
SBATCH_ROOT = GRAPH_ROOT / "sbatch_correct_lag_300"
AUDIT_CSV = GRAPH_ROOT / "correct_lag_300_audit.csv"
SUBMIT_SCRIPT = GRAPH_ROOT / "Ocelote_submit_correct_lag_300.sh"

LOSSES = ("center_stage5", "jkge", "weighted_mse")
MODEL_SETS: list[dict[str, Any]] = [
    {
        "name": "No_Negative/lag14",
        "template_base": PROJECT_ROOT / "configs/extra_seed_450/No_Negative",
        "run_base": RUN_ROOT / "No_Negative/lag14",
        "max_lag": 14,
        "temporal_head": None,
    },
    {
        "name": "lag/lag7",
        "template_base": PROJECT_ROOT / "configs/extra_seed_450/lag/lag7",
        "run_base": RUN_ROOT / "lag/lag7",
        "max_lag": 7,
        "temporal_head": None,
    },
    {
        "name": "lag/lag30",
        "template_base": PROJECT_ROOT / "configs/extra_seed_450/lag/lag30",
        "run_base": RUN_ROOT / "lag/lag30",
        "max_lag": 30,
        "temporal_head": None,
    },
    {
        "name": "lag/no_cnn_lag14",
        "template_base": PROJECT_ROOT / "configs/extra_seed_450/lag/no_cnn",
        "run_base": RUN_ROOT / "lag/no_cnn_lag14",
        "max_lag": 14,
        "temporal_head": "none",
    },
]

VALID_SCALER_FALLBACKS = [
    PROJECT_ROOT / "scalers/noah_post_transfer_runoff_scaler_batch_ablation_weighted_mse_seed777_bs1_tgb64.yml",
]

SEEDS_PER_SCENARIO = 20
SEED_RNG = 20260514
ACCOUNTS = ["andrbenn", "hoshin", "niug", "behrangi"]
JOBS_PER_ACCOUNT = 10
MAX_MODELS_PER_JOB = 6
SBATCH_TIME = "4-00:00:00"


class _QuotedDigitStringDumper(yaml.SafeDumper):
    """Keep gauge IDs quoted so ruamel's safe loader does not parse ints."""


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "'" if value.isdigit() else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_QuotedDigitStringDumper.add_representer(str, _represent_string)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle, Dumper=_QuotedDigitStringDumper, sort_keys=False, default_flow_style=False)


def _contains_post_transfer_stats(path: Path) -> bool:
    try:
        data = _read_yaml(path)
    except Exception:
        return False
    stats = data.get("routing_runoff_stats")
    return isinstance(stats, dict) and "routing_runoff_post_transfer" in stats


def _resolve_path(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _find_valid_scaler(template_config: Path, cfg: dict[str, Any]) -> Path:
    scaler_cfg = cfg.get("scaler")
    if isinstance(scaler_cfg, dict):
        for key in ("path", "file", "scaler_file"):
            path = _resolve_path(scaler_cfg.get(key))
            if path and path.exists() and _contains_post_transfer_stats(path):
                return path

    template_run = Path(cfg.get("run_dir", template_config.parent))
    if not template_run.is_absolute():
        template_run = PROJECT_ROOT / template_run
    for path in sorted((template_run / "scalers").glob("*.yml")):
        if _contains_post_transfer_stats(path):
            return path

    for path in VALID_SCALER_FALLBACKS:
        if path.exists() and _contains_post_transfer_stats(path):
            return path

    raise FileNotFoundError(f"No valid post-transfer scaler found for {template_config}")


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


def _configure_scaler(cfg: dict[str, Any], scaler_path: Path) -> None:
    scaler_cfg = cfg.setdefault("scaler", {})
    if not isinstance(scaler_cfg, dict):
        raise ValueError("Config has non-mapping scaler block")
    scaler_cfg["path"] = str(scaler_path)
    scaler_cfg["save"] = False


def _configure_lags(cfg: dict[str, Any], max_lag: int, temporal_head: str | None) -> None:
    routing_model = cfg.setdefault("routing_model", {})
    if not isinstance(routing_model, dict):
        raise ValueError("Config has non-mapping routing_model block")
    routing_model["runoff_lags"] = list(range(max_lag + 1))
    routing_model["routing_lag_context_days"] = max_lag
    if temporal_head is not None:
        routing_model["temporal_head"] = temporal_head


def _find_template_config(template_base: Path, loss_name: str) -> Path:
    base = template_base / loss_name
    candidates = sorted(base.glob("*.yml"))
    if not candidates:
        raise FileNotFoundError(f"No template configs found under {base}")
    return candidates[0]


def _existing_seeds(*roots: Path) -> set[int]:
    seeds: set[int] = set()
    for root in roots:
        for path in root.rglob("seed*"):
            suffix = path.name.removeprefix("seed")
            if suffix.isdigit():
                seeds.add(int(suffix))
    return seeds


def _make_seed_pool() -> dict[tuple[str, str], list[int]]:
    rng = random.Random(SEED_RNG)
    used = _existing_seeds(PROJECT_ROOT / "runs/correct_lag", PROJECT_ROOT / "configs/correct_lag")
    out: dict[tuple[str, str], list[int]] = {}
    for model_set in MODEL_SETS:
        for loss_name in LOSSES:
            seeds: list[int] = []
            while len(seeds) < SEEDS_PER_SCENARIO:
                seed = rng.randint(100000, 999999)
                if seed in used:
                    continue
                used.add(seed)
                seeds.append(seed)
            out[(model_set["name"], loss_name)] = seeds
    return out


def _create_configs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seed_pool = _make_seed_pool()

    for model_set in MODEL_SETS:
        for loss_name in LOSSES:
            scenario = f"{model_set['name']}/{loss_name}"
            template_config = _find_template_config(model_set["template_base"], loss_name)
            template_cfg = _read_yaml(template_config)
            source_scaler = _find_valid_scaler(template_config, template_cfg)

            for seed in seed_pool[(model_set["name"], loss_name)]:
                run_dir = model_set["run_base"] / loss_name / f"seed{seed}"
                config_path = CONFIG_ROOT / model_set["name"] / loss_name / f"seed{seed}.yml"
                scaler_path = run_dir / "scalers" / "input_scaler.yml"

                cfg = _read_yaml(template_config)
                _set_seed_fields(cfg, seed)
                _set_run_dir_fields(cfg, run_dir)
                _configure_lags(cfg, int(model_set["max_lag"]), model_set["temporal_head"])

                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_scaler, scaler_path)
                _configure_scaler(cfg, scaler_path)

                run_dir.mkdir(parents=True, exist_ok=True)
                _write_yaml(config_path, cfg)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": str(seed),
                        "max_lag": str(model_set["max_lag"]),
                        "runoff_lags": ",".join(str(v) for v in range(int(model_set["max_lag"]) + 1)),
                        "config_path": str(config_path),
                        "run_dir": str(run_dir),
                        "template_config": str(template_config),
                        "scaler_path": str(scaler_path),
                    }
                )

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "seed",
                "max_lag",
                "runoff_lags",
                "config_path",
                "run_dir",
                "template_config",
                "scaler_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _chunk_rows(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    total_jobs = len(ACCOUNTS) * JOBS_PER_ACCOUNT
    index = 0
    for job_index in range(total_jobs):
        remaining_jobs = total_jobs - job_index
        remaining_rows = len(rows) - index
        size = min(MAX_MODELS_PER_JOB, (remaining_rows + remaining_jobs - 1) // remaining_jobs)
        chunks.append(rows[index : index + size])
        index += size
    if index != len(rows):
        raise RuntimeError(f"Chunking mismatch: assigned {index} of {len(rows)}")
    return chunks


def _write_sbatch_scripts(rows: list[dict[str, str]]) -> None:
    SBATCH_ROOT.mkdir(parents=True, exist_ok=True)
    for stale in SBATCH_ROOT.glob("*.sh"):
        stale.unlink()
    chunks = _chunk_rows(rows)
    sbatches: list[Path] = []

    for job_index, chunk in enumerate(chunks, start=1):
        account = ACCOUNTS[(job_index - 1) // JOBS_PER_ACCOUNT]
        script = SBATCH_ROOT / f"Ocelote_correct_lag_300_{job_index:02d}_{account}.sh"
        sbatches.append(script)
        config_lines = "\n".join(f'  "{row["config_path"]}"' for row in chunk)
        script.write_text(
            f"""#!/bin/bash
#SBATCH --job-name=correctlag_{job_index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time={SBATCH_TIME}
#SBATCH --output=/xdisk/tyferre/farmani/Graph_Routing/logs/correctlag_{job_index:02d}_%j.out
#SBATCH --error=/xdisk/tyferre/farmani/Graph_Routing/logs/correctlag_{job_index:02d}_%j.err
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

echo "Starting ${{#CONFIGS[@]}} corrected-lag training runs on one GPU."
for CONFIG in "${{CONFIGS[@]}}"; do
  echo "Training config: $CONFIG"
  python -m lstm_gnn_routing.cli.main train --config-file "$CONFIG"
done
echo "Finished corrected-lag job {job_index:02d}."
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
    print(f"Created {len(rows)} corrected-lag configs.")
    print(f"Audit CSV: {AUDIT_CSV}")
    print(f"Config root: {CONFIG_ROOT}")
    print(f"Run root: {RUN_ROOT}")
    print(f"Sbatch root: {SBATCH_ROOT}")
    print(f"Submit helper: {SUBMIT_SCRIPT}")
    print(f"Accounts: {', '.join(ACCOUNTS)} ({JOBS_PER_ACCOUNT} jobs each)")


if __name__ == "__main__":
    main()
