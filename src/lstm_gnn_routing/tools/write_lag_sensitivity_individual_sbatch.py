"""Write individual sbatch scripts for lag/CNN sensitivity experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
SCRIPT_DIR = PROJECT_ROOT / "sbatch_lag_sensitivity_best4"
LOG_DIR = PROJECT_ROOT / "logs"
COMMON = PROJECT_ROOT / "Ocelote_train_lag_sensitivity_best4_grouped_common.sh"

ACCOUNTS = ["behrangi", "niug", "andrbenn", "hoshin", "tyferre"]
GROUP_SIZE = 3
TOTAL_MODELS = 160
TOTAL_GROUPS = (TOTAL_MODELS + GROUP_SIZE - 1) // GROUP_SIZE


def _script_text(group_index: int, account: str, local_task_id: int) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=lag4x3_{group_index:03d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output={LOG_DIR}/lag4x3_individual_{group_index:03d}_%j.out
#SBATCH --error={LOG_DIR}/lag4x3_individual_{group_index:03d}_%j.err

mkdir -p {LOG_DIR}
export ACCOUNT_FILTER="{account}"
export GROUP_SIZE="{GROUP_SIZE}"
export SLURM_ARRAY_TASK_ID="{local_task_id}"
bash {COMMON}
"""


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    account_group_counts = {account: 0 for account in ACCOUNTS}
    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for group_index in range(TOTAL_GROUPS):
        account = ACCOUNTS[group_index % len(ACCOUNTS)]
        local_task_id = account_group_counts[account]
        account_group_counts[account] += 1

        script_path = SCRIPT_DIR / f"Ocelote_train_lag_sensitivity_best4_group{group_index:03d}_{account}.sh"
        script_path.write_text(_script_text(group_index, account, local_task_id), encoding="utf-8")
        submit_lines.append(f"sbatch {script_path}")

    submit_path = PROJECT_ROOT / "Ocelote_submit_train_lag_sensitivity_best4_individual.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")

    print(f"Wrote {TOTAL_GROUPS} individual sbatch files to {SCRIPT_DIR}")
    print(f"Wrote submit helper to {submit_path}")
    print("Groups per account:")
    for account, count in account_group_counts.items():
        print(f"  {account}: {count}")


if __name__ == "__main__":
    main()
