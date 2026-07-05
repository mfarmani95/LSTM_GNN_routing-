"""Write individual grouped sbatch files for the 120 graph-model runs."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
REPO_ROOT = PROJECT_ROOT / "LSTM_GNN_routing-"
SCRIPT_DIR = PROJECT_ROOT / "sbatch_graph_models_full"
LOG_DIR = PROJECT_ROOT / "logs"
MANIFEST = REPO_ROOT / "configs/graph_models/graph_model_full_manifest.csv"

GROUP3_ACCOUNTS = ["niug", "andrbenn", "behrangi"]
GROUP2_ACCOUNTS = ["tyferre", "hoshin"]


def _script_text(group_index: int, account: str, rows: list[dict[str, str]], hours: int) -> str:
    commands = []
    for row in rows:
        commands.append(
            "\n".join(
                [
                    f"echo \"Training {row['graph_model']} {row['candidate']} seed{row['seed']}\"",
                    f"python -m lstm_gnn_routing.cli.main train --config-file {row['config_file']}",
                ]
            )
        )
    body = "\n\n".join(commands)
    models_label = "_".join(f"{row['graph_model']}-{row['candidate']}-s{row['seed']}" for row in rows)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gmfull_{group_index:03d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time={hours}:00:00
#SBATCH --output={LOG_DIR}/graph_models_full_{group_index:03d}_%j.out
#SBATCH --error={LOG_DIR}/graph_models_full_{group_index:03d}_%j.err

set -euo pipefail

module purge
module load gnu8
module load python/3.11
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate

cd {REPO_ROOT}
echo "Grouped graph-model job {group_index:03d} on account {account}"
echo "Models: {models_label}"

{body}
"""


def _chunks(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8")))
    if len(rows) != 120:
        raise ValueError(f"Expected 120 graph-model rows, found {len(rows)}")

    first_90 = rows[:90]
    final_30 = rows[90:]
    groups: list[tuple[str, list[dict[str, str]], int]] = []

    for index, group_rows in enumerate(_chunks(first_90, 3)):
        groups.append((GROUP3_ACCOUNTS[index % len(GROUP3_ACCOUNTS)], group_rows, 36))

    for index, group_rows in enumerate(_chunks(final_30, 2)):
        groups.append((GROUP2_ACCOUNTS[index % len(GROUP2_ACCOUNTS)], group_rows, 24))

    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    account_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for group_index, (account, group_rows, hours) in enumerate(groups):
        account_counts[account] = account_counts.get(account, 0) + 1
        model_counts[account] = model_counts.get(account, 0) + len(group_rows)
        script_path = SCRIPT_DIR / f"Ocelote_train_graph_models_full_group{group_index:03d}_{account}.sh"
        script_path.write_text(_script_text(group_index, account, group_rows, hours), encoding="utf-8")
        submit_lines.append(f"sbatch {script_path}")

    submit_path = PROJECT_ROOT / "Ocelote_submit_train_graph_models_full.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(groups)} grouped sbatch scripts to {SCRIPT_DIR}")
    print(f"Wrote submit helper: {submit_path}")
    print("Job groups per account:")
    for account in GROUP3_ACCOUNTS + GROUP2_ACCOUNTS:
        print(f"  {account}: {account_counts.get(account, 0)} jobs, {model_counts.get(account, 0)} models")


if __name__ == "__main__":
    main()
