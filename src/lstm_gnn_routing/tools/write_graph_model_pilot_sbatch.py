"""Write individual sbatch files for graph-model seed-42 pilot runs."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
REPO_ROOT = PROJECT_ROOT / "LSTM_GNN_routing-"
SCRIPT_DIR = PROJECT_ROOT / "sbatch_graph_model_pilot"
LOG_DIR = PROJECT_ROOT / "logs"
MANIFEST = REPO_ROOT / "configs/graph_model_pilot/graph_model_pilot_manifest.csv"
ACCOUNTS = ["niug", "andrbenn", "tyferre", "hoshin"]


def _script_text(row: dict[str, str], account: str, index: int) -> str:
    graph_model = row["graph_model"]
    candidate = row["candidate"]
    config_file = row["config_file"]
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gm_{graph_model[:4]}_{candidate[:4]}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output={LOG_DIR}/graph_model_pilot_{index:02d}_{graph_model}_{candidate}_%j.out
#SBATCH --error={LOG_DIR}/graph_model_pilot_{index:02d}_{graph_model}_{candidate}_%j.err

set -euo pipefail

module purge
module load gnu8
module load python/3.11
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate

cd {REPO_ROOT}
python -m lstm_gnn_routing.cli.main train --config-file {config_file}
"""


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8")))

    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for index, row in enumerate(rows):
        account = ACCOUNTS[index % len(ACCOUNTS)]
        script_path = (
            SCRIPT_DIR
            / f"Ocelote_train_graph_model_pilot_{index:02d}_{row['graph_model']}_{row['candidate']}.sh"
        )
        script_path.write_text(_script_text(row, account, index), encoding="utf-8")
        submit_lines.append(f"sbatch {script_path}")

    submit_path = PROJECT_ROOT / "Ocelote_submit_train_graph_model_pilot_seed42.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} pilot sbatch scripts to {SCRIPT_DIR}")
    print(f"Wrote submit helper: {submit_path}")


if __name__ == "__main__":
    main()
