"""Generate JKGE_SA_V1 configs with fixed non-overlapping sections."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

from lstm_gnn_routing.tools.prepare_lag_sensitivity_config import (
    _QuotedNumericStringDumper,
    _load_yaml,
    _stringify_mapping_keys,
)


REPO_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
SOURCE_MANIFEST = REPO_ROOT / "configs/jkge_sa_v2/jkge_sa_v2_manifest.csv"


def _find_valid_reference_scaler() -> Path:
    for path in sorted((REPO_ROOT / "scalers").glob("noah_post_transfer_runoff_scaler_no_negative_jkge_split619873_seed*.yml")):
        data = _load_yaml(path)
        if (
            "routing_runoff_post_transfer" in (data.get("routing_runoff_stats") or {})
            and "streamflow_percentile_minmax_p01_p99" in (data.get("target_stats") or {})
        ):
            return path
    for path in sorted((REPO_ROOT / "scalers").glob("*.yml")):
        data = _load_yaml(path)
        if (
            "routing_runoff_post_transfer" in (data.get("routing_runoff_stats") or {})
            and "streamflow_percentile_minmax_p01_p99" in (data.get("target_stats") or {})
        ):
            return path
    raise FileNotFoundError("Could not find a scaler with both routing_runoff_post_transfer and target stats.")


def _install_private_scaler(cfg: dict, run_dir: Path, reference_scaler: Path) -> None:
    scaler_path = run_dir / "scalers" / "input_scaler.yml"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_scaler, scaler_path)
    cfg["scaler"] = {
        "path": str(scaler_path),
        "save": False,
    }


def main() -> None:
    rows = list(csv.DictReader(SOURCE_MANIFEST.open("r", encoding="utf-8")))
    manifest_rows = ["family,scenario,graph_model,seed,source_config,config_file,run_dir"]
    reference_scaler = _find_valid_reference_scaler()
    print(f"Using reference scaler: {reference_scaler}")

    for row in rows:
        source_config = Path(row["config_file"])
        cfg = _stringify_mapping_keys(_load_yaml(source_config))
        training = cfg.setdefault("training", {})
        training["loss"] = "jkge_sa_v1"
        training["jkge_sa_v1_benchmark"] = "non_overlapping_sections"
        training["jkge_sa_v1_section_length"] = 31
        training["jkge_sa_v1_eps"] = 1.0e-8
        training["jkge_sa_v1_min_valid_sections"] = 1
        training.pop("jkge_sa_v2_benchmark", None)
        training.pop("jkge_sa_v2_section_length", None)
        training.pop("jkge_sa_v2_eps", None)
        training.pop("jkge_sa_v2_min_valid_sections", None)
        training.pop("jkge_ma_v1_window_length", None)
        training.pop("jkge_ma_v1_eps", None)

        old_run_dir = str(cfg["run_dir"])
        run_dir = Path(old_run_dir.replace("/runs/jkge_sa_v2/", "/runs/jkge_sa_v1/"))
        cfg["run_dir"] = str(run_dir)
        cfg["experiment_name"] = str(cfg.get("experiment_name", "")).replace("jkge_sa_v2", "jkge_sa_v1")
        _install_private_scaler(cfg, run_dir, reference_scaler)

        config_file = Path(
            str(source_config)
            .replace("/configs/jkge_sa_v2/", "/configs/jkge_sa_v1/")
            .replace("jkge_sa_v2_", "jkge_sa_v1_")
        )
        config_file.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with config_file.open("w", encoding="utf-8") as fp:
            yaml.dump(
                cfg,
                fp,
                Dumper=_QuotedNumericStringDumper,
                sort_keys=False,
                default_flow_style=False,
            )

        manifest_rows.append(
            f"{row['family']},{row['scenario']},{row['graph_model']},{row['seed']},{source_config},{config_file},{run_dir}"
        )

    manifest = REPO_ROOT / "configs/jkge_sa_v1/jkge_sa_v1_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} JKGE_SA_V1 configs")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
