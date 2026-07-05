"""Generate seed-42 pilot configs for alternative graph convolution models."""

from __future__ import annotations

from pathlib import Path

import yaml

from lstm_gnn_routing.tools.prepare_lag_sensitivity_config import (
    _QuotedNumericStringDumper,
    _copy_loss_settings,
    _copy_split_settings,
    _copy_supervision_settings,
    _load_yaml,
    _set_seed,
    _stringify_mapping_keys,
)


REPO_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")
REFERENCE_SPLIT_CONFIG = REPO_ROOT / "runs/No_Negative/weighted_mse/seed615222/config.yml"

GRAPH_MODELS = [
    ("gcn", "GCN"),
    ("graphsage", "GraphSAGE"),
    ("gineconv", "GINEConv"),
]
CANDIDATES = ["center_stage5", "jkge", "weighted_mse", "fractional_area"]
SEED = 42


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


def _set_graph_model(cfg: dict, conv_type: str) -> None:
    routing_model = cfg.setdefault("routing_model", {})
    if not isinstance(routing_model, dict):
        raise ValueError("routing_model must be a YAML mapping")
    routing_model["conv_type"] = conv_type
    if conv_type.lower() in {"gineconv", "gine"}:
        routing_model["edge_attr_key"] = "edge_attr"


def main() -> None:
    split_cfg = _load_yaml(REFERENCE_SPLIT_CONFIG)
    loss_cfg = _load_yaml(_center_stage5_loss_config())
    manifest_rows = ["graph_model,candidate,seed,config_file,run_dir,source_config"]

    for graph_slug, conv_type in GRAPH_MODELS:
        for candidate in CANDIDATES:
            source_config = _first_candidate_config(candidate)
            cfg = _stringify_mapping_keys(_load_yaml(source_config))
            _copy_split_settings(cfg, split_cfg)
            _copy_loss_settings(cfg, loss_cfg)
            _copy_supervision_settings(cfg, split_cfg)
            _set_seed(cfg, SEED)
            _set_graph_model(cfg, conv_type)

            run_dir = REPO_ROOT / "runs/graph_model_pilot" / graph_slug / candidate / f"seed{SEED}"
            config_file = (
                REPO_ROOT
                / "configs/graph_model_pilot"
                / graph_slug
                / candidate
                / f"noah_precomputed_runoff_gnn_curriculum_{graph_slug}_{candidate}_seed{SEED}.yml"
            )
            cfg["experiment_name"] = f"graph_model_pilot_{graph_slug}_{candidate}_seed{SEED}"
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

            manifest_rows.append(f"{graph_slug},{candidate},{SEED},{config_file},{run_dir},{source_config}")

    manifest = REPO_ROOT / "configs/graph_model_pilot/graph_model_pilot_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_rows) - 1} graph-model pilot configs")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
