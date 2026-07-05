#!/usr/bin/env python3
"""Create sbatch jobs to evaluate and RAPID-compare the extra_seed_450 runs."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
GRAPH_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing")
AUDIT_CSV = GRAPH_ROOT / "extra_seed_450_audit.csv"
SBATCH_ROOT = GRAPH_ROOT / "sbatch_extra_seed_450_eval_plot"
SUBMIT_SCRIPT = GRAPH_ROOT / "Ocelote_evaluate_and_plot_extra_seed_450.sh"

ACCOUNTS = ["andrbenn", "hoshin", "niug", "behrangi"]
JOBS_PER_ACCOUNT = 10
MAX_RUNS_PER_JOB = 12
SBATCH_TIME = "3-00:00:00"


def _read_run_dirs() -> list[Path]:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(AUDIT_CSV)
    seen: set[str] = set()
    run_dirs: list[Path] = []
    with AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("run_dir", "").strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            run_dirs.append(Path(raw))
    return run_dirs


def _chunk(items: list[Path]) -> list[list[Path]]:
    total_jobs = len(ACCOUNTS) * JOBS_PER_ACCOUNT
    chunks: list[list[Path]] = []
    index = 0
    for job_index in range(total_jobs):
        remaining_jobs = total_jobs - job_index
        remaining_items = len(items) - index
        size = min(MAX_RUNS_PER_JOB, (remaining_items + remaining_jobs - 1) // remaining_jobs)
        chunks.append(items[index : index + size])
        index += size
    if index != len(items):
        raise RuntimeError(f"Chunking mismatch: assigned {index} of {len(items)}")
    return chunks


def _write_sbatch_scripts(run_dirs: list[Path]) -> None:
    SBATCH_ROOT.mkdir(parents=True, exist_ok=True)
    for stale in SBATCH_ROOT.glob("*.sh"):
        stale.unlink()

    chunks = _chunk(run_dirs)
    sbatches: list[Path] = []
    for job_index, chunk in enumerate(chunks, start=1):
        account = ACCOUNTS[(job_index - 1) // JOBS_PER_ACCOUNT]
        script = SBATCH_ROOT / f"Ocelote_extra_seed_450_eval_plot_{job_index:02d}_{account}.sh"
        sbatches.append(script)
        run_lines = "\n".join(f'  "{run_dir}"' for run_dir in chunk)
        script.write_text(
            f"""#!/bin/bash
#SBATCH --job-name=evalplot450_{job_index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time={SBATCH_TIME}
#SBATCH --output=/xdisk/tyferre/farmani/Graph_Routing/logs/evalplot450_{job_index:02d}_%j.out
#SBATCH --error=/xdisk/tyferre/farmani/Graph_Routing/logs/evalplot450_{job_index:02d}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=farmani@arizona.edu

set -euo pipefail

module purge
module load gnu8
module load python/3.11

PROJECT_ROOT="/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-"
GRAPH_ROOT="/xdisk/tyferre/farmani/Graph_Routing"
cd "$PROJECT_ROOT"

mkdir -p "$GRAPH_ROOT/logs"
source /xdisk/tyferre/farmani/env/noahvec_uv/bin/activate
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/python/3.11.4/lib:${{LD_LIBRARY_PATH:-}}
export MPLCONFIGDIR=/xdisk/tyferre/farmani/env/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

PERIOD="${{PERIOD:-test}}"
FORCE_EVALUATE="${{FORCE_EVALUATE:-0}}"
FORCE_ANALYZE="${{FORCE_ANALYZE:-0}}"
GAUGE_METADATA="$PROJECT_ROOT/data/streamflow/30_gauges_IN_LAMBERT.csv"
COMID_CACHE="$PROJECT_ROOT/data/streamflow/usgs_nldi_comids.csv"
BACKGROUND_SHAPEFILE="$PROJECT_ROOT/data/HUC4"

RAPID_ARG=""
if [[ -n "${{RAPID_FILE:-}}" ]]; then
  RAPID_ARG="--rapid-file $RAPID_FILE"
fi

RUN_DIRS=(
{run_lines}
)

echo "Starting eval+RAPID-plot job {job_index:02d} with ${{#RUN_DIRS[@]}} runs."

for RUN_DIR in "${{RUN_DIRS[@]}}"; do
  echo "============================================================"
  echo "Run: $RUN_DIR"

  if [[ ! -f "$RUN_DIR/config.yml" ]]; then
    echo "Skipping missing config.yml: $RUN_DIR"
    continue
  fi
  if [[ ! -f "$RUN_DIR/training_history.csv" ]]; then
    echo "Skipping incomplete run with no training_history.csv: $RUN_DIR"
    continue
  fi

  OUT_DIR="$RUN_DIR/evaluation_${{PERIOD}}_best_final_stage_model"
  TIMESERIES="$OUT_DIR/${{PERIOD}}_timeseries.csv"

  if [[ "$FORCE_EVALUATE" == "1" || ! -f "$TIMESERIES" ]]; then
    CHECKPOINT_FILE="$(RUN_DIR="$RUN_DIR" python - <<'PY'
from pathlib import Path
import os

run = Path(os.environ["RUN_DIR"])
priority = [
    run / "best_final_stage_model.pt",
    run / "best_model.pt",
    run / "last_model.pt",
]
for path in priority:
    if path.exists():
        print(path)
        raise SystemExit

stage_best = sorted(run.glob("best_stage_*.pt"))
if stage_best:
    print(stage_best[-1])
    raise SystemExit

all_ckpts = sorted(list(run.rglob("*.pt")) + list(run.rglob("*.pth")), key=lambda p: p.stat().st_mtime)
if all_ckpts:
    print(all_ckpts[-1])
PY
)"
    if [[ -z "$CHECKPOINT_FILE" || ! -f "$CHECKPOINT_FILE" ]]; then
      echo "Skipping run with no checkpoint: $RUN_DIR"
      continue
    fi

    echo "Evaluating with checkpoint: $CHECKPOINT_FILE"
    if ! python -m lstm_gnn_routing.cli.main evaluate \\
      --run-dir "$RUN_DIR" \\
      --checkpoint-file "$CHECKPOINT_FILE" \\
      --period "$PERIOD" \\
      --output-dir "$OUT_DIR"; then
      echo "Evaluation failed, continuing to next run: $RUN_DIR"
      continue
    fi
  else
    echo "Evaluation already exists: $TIMESERIES"
  fi

  if [[ ! -f "$TIMESERIES" ]]; then
    echo "Skipping RAPID comparison because timeseries is missing: $TIMESERIES"
    continue
  fi

  RAPID_OUT="$OUT_DIR/rapid_comparison"
  if [[ "$FORCE_ANALYZE" != "1" && -d "$RAPID_OUT/plots" ]]; then
    echo "RAPID comparison already exists: $RAPID_OUT"
    continue
  fi

  echo "Plotting RAPID comparison: $OUT_DIR"
  if ! python -m lstm_gnn_routing.tools.analyze_rapid_vs_gnn \\
    --evaluation-dir "$OUT_DIR" \\
    $RAPID_ARG \\
    --gauge-metadata "$GAUGE_METADATA" \\
    --nldi-comid-cache "$COMID_CACHE" \\
    --infer-comids-from-nldi \\
    --background-shapefile "$BACKGROUND_SHAPEFILE" \\
    --stations dam_filtered \\
    --output-dir "$RAPID_OUT"; then
    echo "RAPID comparison failed, continuing to next run: $RUN_DIR"
    continue
  fi
done

echo "Finished eval+RAPID-plot job {job_index:02d}."
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
    run_dirs = _read_run_dirs()
    _write_sbatch_scripts(run_dirs)
    print(f"Prepared eval+plot jobs for {len(run_dirs)} run directories.")
    print(f"Sbatch root: {SBATCH_ROOT}")
    print(f"Submit helper: {SUBMIT_SCRIPT}")
    print(f"Jobs: {len(list(SBATCH_ROOT.glob('*.sh')))}")
    print(f"Accounts: {', '.join(ACCOUNTS)} ({JOBS_PER_ACCOUNT} jobs each)")


if __name__ == "__main__":
    main()
