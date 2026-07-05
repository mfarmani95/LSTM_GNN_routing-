"""Generate all lag/CNN sensitivity YAML configs before Slurm submission."""

from __future__ import annotations

from pathlib import Path

import yaml

from lstm_gnn_routing.tools.prepare_lag_sensitivity_config import (
    _QuotedNumericStringDumper,
    _copy_loss_settings,
    _copy_split_settings,
    _copy_supervision_settings,
    _load_yaml,
    _parse_lags,
    _set_routing_options,
    _set_seed,
    _stringify_mapping_keys,
)


REPO_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")

ACCOUNTS = ["behrangi", "niug", "andrbenn", "hoshin", "tyferre"]
SCENARIOS = [
    ("lag3", "0,1,2,3", 3, "conv1d"),
    ("lag7", "0,1,2,3,7", 7, "conv1d"),
    ("lag30", "0,1,2,3,7,14,30", 30, "conv1d"),
    ("no_cnn", "0,1,2,3,7,14", 14, "none"),
]
CANDIDATES = ["center_stage5", "fractional_area", "jkge", "weighted_mse"]
REPS = 10

REFERENCE_SPLIT_CONFIG = REPO_ROOT / "runs/No_Negative/weighted_mse/seed615222/config.yml"


def _first_candidate_config(candidate: str) -> Path:
    if candidate == "weighted_mse":
        return REFERENCE_SPLIT_CONFIG
    matches = sorted((REPO_ROOT / "runs/No_Negative" / candidate).glob("seed*/config.yml"))
    if not matches:
        raise FileNotFoundError(f"No source config found for candidate {candidate}")
    return matches[0]


def _center_stage5_loss_config() -> Path:
    matches = sorted((REPO_ROOT / "runs/No_Negative/center_stage5").glob("seed*/config.yml"))
    if not matches:
        raise FileNotFoundError("No center_stage5 config found for loss reference")
    return matches[0]


def _seed(global_index: int, scenario_index: int, candidate_index: int, rep_index: int) -> int:
    return (
        711001
        + global_index * 104729
        + scenario_index * 1009
        + candidate_index * 9176
        + rep_index * 7919
    ) % 900000 + 100000


def main() -> None:
    split_cfg = _load_yaml(REFERENCE_SPLIT_CONFIG)
    loss_cfg = _load_yaml(_center_stage5_loss_config())
    manifest_rows = [
        "global_index,account,scenario,candidate,seed,config_file,run_dir",
    ]

    count = 0
    global_index = 0
    for scenario_index, (scenario, raw_lags, context_days, temporal_head) in enumerate(SCENARIOS):
        for candidate_index, candidate in enumerate(CANDIDATES):
            source_config = _first_candidate_config(candidate)
            for rep_index in range(REPS):
                account = ACCOUNTS[global_index % len(ACCOUNTS)]
                seed = _seed(global_index, scenario_index, candidate_index, rep_index)
                run_dir = REPO_ROOT / "runs/lag" / scenario / candidate / f"seed{seed}"
                config_file = (
                    REPO_ROOT
                    / "configs/lag"
                    / scenario
                    / candidate
                    / f"noah_precomputed_runoff_gnn_curriculum_{scenario}_{candidate}_seed{seed}.yml"
                )
                experiment_name = f"lag_{scenario}_{candidate}_seed{seed}"

                cfg = _stringify_mapping_keys(_load_yaml(source_config))
                _copy_split_settings(cfg, split_cfg)
                _copy_loss_settings(cfg, loss_cfg)
                _copy_supervision_settings(cfg, split_cfg)
                _set_seed(cfg, seed)
                _set_routing_options(cfg, _parse_lags(raw_lags), context_days, temporal_head)
                cfg["experiment_name"] = experiment_name
                cfg["run_dir"] = str(run_dir)

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
                    f"{global_index},{account},{scenario},{candidate},{seed},{config_file},{run_dir}"
                )
                count += 1
                global_index += 1

    manifest = REPO_ROOT / "configs/lag/lag_sensitivity_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote {count} configs")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
