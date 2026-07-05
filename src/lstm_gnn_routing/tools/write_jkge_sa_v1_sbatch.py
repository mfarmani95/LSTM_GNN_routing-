"""Write non-array sbatch scripts for JKGE_SA_V1 reruns."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
REPO_ROOT = PROJECT_ROOT / "LSTM_GNN_routing-"
MANIFEST = REPO_ROOT / "configs/jkge_sa_v1/jkge_sa_v1_manifest.csv"
SCRIPT_DIR = PROJECT_ROOT / "sbatch_jkge_sa_v1"
LOG_DIR = PROJECT_ROOT / "logs"
ACCOUNTS = ["niug", "andrbenn", "behrangi", "tyferre", "hoshin"]
JOBS_PER_ACCOUNT = 10


def _script_text(account: str, job_index: int, rows: list[dict[str, str]]) -> str:
    commands = []
    for row in rows:
        label = "/".join(
            item
            for item in [
                row["family"],
                row["scenario"],
                row["graph_model"],
                f"seed{row['seed']}",
            ]
            if item
        )
        commands.append(
            "\n".join(
                [
                    f"echo \"Training JKGE_SA_V1 {label}\"",
                    f"python -m lstm_gnn_routing.cli.main train --config-file {row['config_file']}",
                ]
            )
        )
    body = "\n\n".join(commands)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=jsa1_{account[:3]}_{job_index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output={LOG_DIR}/jkge_sa_v1_{account}_{job_index:02d}_%j.out
#SBATCH --error={LOG_DIR}/jkge_sa_v1_{account}_{job_index:02d}_%j.err

set -euo pipefail

module purge
module load gnu8
module load python/3.11
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate

cd {REPO_ROOT}
echo "Starting JKGE_SA_V1 grouped job {job_index:02d} for account {account}"
echo "Model count: {len(rows)}"

{body}
"""


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8")))
    if len(rows) != 80:
        raise ValueError(f"Expected 80 JKGE_SA_V1 configs, found {len(rows)}")

    grouped = {(account, job_index): [] for account in ACCOUNTS for job_index in range(JOBS_PER_ACCOUNT)}
    for idx, row in enumerate(rows):
        account = ACCOUNTS[idx % len(ACCOUNTS)]
        job_index = (idx // len(ACCOUNTS)) % JOBS_PER_ACCOUNT
        grouped[(account, job_index)].append(row)

    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for account in ACCOUNTS:
        for job_index in range(JOBS_PER_ACCOUNT):
            job_rows = grouped[(account, job_index)]
            if not job_rows:
                continue
            script_path = SCRIPT_DIR / f"Ocelote_train_jkge_sa_v1_{account}_{job_index:02d}.sh"
            script_path.write_text(_script_text(account, job_index, job_rows), encoding="utf-8")
            submit_lines.append(f"sbatch {script_path}")

    submit_path = PROJECT_ROOT / "Ocelote_submit_train_jkge_sa_v1.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(submit_lines) - 3} sbatch scripts to {SCRIPT_DIR}")
    print(f"Wrote submit helper: {submit_path}")
    for account in ACCOUNTS:
        model_count = sum(len(grouped[(account, job_index)]) for job_index in range(JOBS_PER_ACCOUNT))
        job_count = sum(1 for job_index in range(JOBS_PER_ACCOUNT) if grouped[(account, job_index)])
        print(f"  {account}: {job_count} jobs, {model_count} models")


if __name__ == "__main__":
    main()
