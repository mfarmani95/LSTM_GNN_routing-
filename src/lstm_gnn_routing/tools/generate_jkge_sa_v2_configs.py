"""Generate JKGE_SA_V2 rerun configs from previous JKGE experiments."""

from __future__ import annotations

from pathlib import Path
import shutil

import yaml

from lstm_gnn_routing.tools.prepare_lag_sensitivity_config import (
    _QuotedNumericStringDumper,
    _load_yaml,
    _stringify_mapping_keys,
)


REPO_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")


def _set_jkge_sa_v2_training(cfg: dict) -> None:
    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be a YAML mapping")

    training["loss"] = "jkge_sa_v2"
    training["jkge_sa_v2_benchmark"] = "non_overlapping_sections"
    training["jkge_sa_v2_section_length"] = 30
    training["jkge_sa_v2_eps"] = 1.0e-8
    training["jkge_sa_v2_min_valid_sections"] = 1
    training["early_stopping_patience"] = 50
    # These belong to the older moving-average JKGE loss. Leaving them in the
    # config is confusing and can make a quick inspection look wrong.
    training.pop("jkge_benchmark", None)
    training.pop("jkge_window", None)
    training.pop("jkge_eps", None)


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


def _add_stage_epochs(cfg: dict, extra_epochs: int = 50) -> None:
    curriculum = cfg.get("curriculum")
    if not isinstance(curriculum, dict):
        return
    stages = curriculum.get("stages")
    if not isinstance(stages, list):
        return
    for stage in stages:
        if isinstance(stage, dict) and "epochs" in stage:
            stage["epochs"] = int(stage["epochs"]) + int(extra_epochs)


def _write_config(source: Path, output_config: Path, run_dir: Path, experiment_name: str, reference_scaler: Path) -> None:
    cfg = _stringify_mapping_keys(_load_yaml(source))
    _set_jkge_sa_v2_training(cfg)
    _add_stage_epochs(cfg, extra_epochs=50)
    cfg["run_dir"] = str(run_dir)
    cfg["experiment_name"] = experiment_name
    _install_private_scaler(cfg, run_dir, reference_scaler)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as fp:
        yaml.dump(
            cfg,
            fp,
            Dumper=_QuotedNumericStringDumper,
            sort_keys=False,
            default_flow_style=False,
        )


def _seed_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("seed"):
            return part.replace("seed", "", 1)
    raise ValueError(f"Could not infer seed from {path}")


def main() -> None:
    manifest_rows = ["family,scenario,graph_model,seed,source_config,config_file,run_dir"]
    count = 0
    reference_scaler = _find_valid_reference_scaler()
    print(f"Using reference scaler: {reference_scaler}")

    # 1) No_Negative JKGE: 10 seed configs.
    for source in sorted((REPO_ROOT / "runs/No_Negative/jkge").glob("seed*/config.yml")):
        seed = _seed_from_path(source)
        run_dir = REPO_ROOT / "runs/jkge_sa_v2/No_Negative/jkge" / f"seed{seed}"
        config_file = REPO_ROOT / "configs/jkge_sa_v2/No_Negative/jkge" / f"jkge_sa_v2_no_negative_jkge_seed{seed}.yml"
        _write_config(source, config_file, run_dir, f"jkge_sa_v2_no_negative_jkge_seed{seed}", reference_scaler)
        manifest_rows.append(f"No_Negative,jkge,,{seed},{source},{config_file},{run_dir}")
        count += 1

    # 2) lag JKGE: 4 lag scenarios x 10 seeds.
    for source in sorted((REPO_ROOT / "runs/lag").glob("*/jkge/seed*/config.yml")):
        seed = _seed_from_path(source)
        scenario = source.parts[-4]
        run_dir = REPO_ROOT / "runs/jkge_sa_v2/lag" / scenario / "jkge" / f"seed{seed}"
        config_file = (
            REPO_ROOT
            / "configs/jkge_sa_v2/lag"
            / scenario
            / "jkge"
            / f"jkge_sa_v2_lag_{scenario}_jkge_seed{seed}.yml"
        )
        _write_config(source, config_file, run_dir, f"jkge_sa_v2_lag_{scenario}_jkge_seed{seed}", reference_scaler)
        manifest_rows.append(f"lag,{scenario},,{seed},{source},{config_file},{run_dir}")
        count += 1

    # 3) graph_model_pilot JKGE: use seed42 source structure to create 10 seeds
    # per graph model, so this family is also comparable as 10-seed runs.
    pilot_sources = sorted((REPO_ROOT / "runs/graph_model_pilot").glob("*/jkge/seed*/config.yml"))
    graph_model_seeds = [135791, 246802, 357913, 468024, 579135, 680246, 791357, 802468, 913579, 124680]
    for source in pilot_sources:
        graph_model = source.parts[-4]
        for seed in graph_model_seeds:
            run_dir = REPO_ROOT / "runs/jkge_sa_v2/graph_model_pilot" / graph_model / "jkge" / f"seed{seed}"
            config_file = (
                REPO_ROOT
                / "configs/jkge_sa_v2/graph_model_pilot"
                / graph_model
                / "jkge"
                / f"jkge_sa_v2_graph_model_pilot_{graph_model}_jkge_seed{seed}.yml"
            )
            _write_config(
                source,
                config_file,
                run_dir,
                f"jkge_sa_v2_graph_model_pilot_{graph_model}_jkge_seed{seed}",
                reference_scaler,
            )
            # Override seed because the pilot template is seed42.
            cfg = _stringify_mapping_keys(_load_yaml(config_file))
            cfg["seed"] = seed
            cfg["random_seed"] = seed
            cfg.setdefault("training", {})["seed"] = seed
            with config_file.open("w", encoding="utf-8") as fp:
                yaml.dump(
                    cfg,
                    fp,
                    Dumper=_QuotedNumericStringDumper,
                    sort_keys=False,
                    default_flow_style=False,
                )
            manifest_rows.append(f"graph_model_pilot,,{graph_model},{seed},{source},{config_file},{run_dir}")
            count += 1

    manifest = REPO_ROOT / "configs/jkge_sa_v2/jkge_sa_v2_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote {count} JKGE_SA_V2 configs")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
