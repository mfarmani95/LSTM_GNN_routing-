from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path("/xdisk/tyferre/farmani/Graph_Routing/LSTM_GNN_routing-")

SCENARIO_BEST_MODELS: dict[str, dict[str, str]] = {
    "lag3_wmse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag3/weighted_mse"),
        "seed": "seed380399",
    },
    "lag7_wmse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag7/weighted_mse"),
        "seed": "seed746456",
    },
    "lag14_wmse": {
        "base_dir": str(PROJECT_ROOT / "runs/No_Negative/weighted_mse"),
        "seed": "seed505321",
    },
    "lag30_wmse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag30/weighted_mse"),
        "seed": "seed692714",
    },
    "lag14_no_cnn_wmse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/no_cnn/weighted_mse"),
        "seed": "seed463554",
    },
    "lag3_mse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag3/center_stage5"),
        "seed": "seed923649",
    },
    "lag7_mse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag7/center_stage5"),
        "seed": "seed277058",
    },
    "lag14_mse": {
        "base_dir": str(PROJECT_ROOT / "runs/No_Negative/center_stage5"),
        "seed": "seed472930",
    },
    "lag30_mse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag30/center_stage5"),
        "seed": "seed754579",
    },
    "lag14_no_cnn_mse": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/no_cnn/center_stage5"),
        "seed": "seed332100",
    },
    "lag3_jkge": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag3/jkge"),
        "seed": "seed225117",
    },
    "lag7_jkge": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag7/jkge"),
        "seed": "seed252046",
    },
    "lag14_jkge": {
        "base_dir": str(PROJECT_ROOT / "runs/No_Negative/jkge"),
        "seed": "seed286876",
    },
    "lag30_jkge": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/lag30/jkge"),
        "seed": "seed504271",
    },
    "lag14_no_cnn_jkge": {
        "base_dir": str(PROJECT_ROOT / "runs/lag/no_cnn/jkge"),
        "seed": "seed194440",
    },
}

ORDERED_SCENARIOS = [
    "lag3_mse",
    "lag7_mse",
    "lag14_mse",
    "lag30_mse",
    "lag14_no_cnn_mse",
    "lag3_wmse",
    "lag7_wmse",
    "lag14_wmse",
    "lag30_wmse",
    "lag14_no_cnn_wmse",
    "lag3_jkge",
    "lag7_jkge",
    "lag14_jkge",
    "lag30_jkge",
    "lag14_no_cnn_jkge",
]

SCENARIO_LABELS = {
    "lag3_mse": "Lag3",
    "lag7_mse": "Lag7",
    "lag14_mse": "Lag14",
    "lag30_mse": "Lag30",
    "lag14_no_cnn_mse": "No CNN",
    "lag3_wmse": "Lag3",
    "lag7_wmse": "Lag7",
    "lag14_wmse": "Lag14",
    "lag30_wmse": "Lag30",
    "lag14_no_cnn_wmse": "No CNN",
    "lag3_jkge": "Lag3",
    "lag7_jkge": "Lag7",
    "lag14_jkge": "Lag14",
    "lag30_jkge": "Lag30",
    "lag14_no_cnn_jkge": "No CNN",
}


def infer_loss_type(scenario_name: str) -> str:
    if scenario_name.endswith("_wmse"):
        return "wmse"
    if scenario_name.endswith("_jkge"):
        return "jkge"
    return "mse"


def infer_lag_days(scenario_name: str) -> int | None:
    if scenario_name.startswith("lag3"):
        return 3
    if scenario_name.startswith("lag7"):
        return 7
    if scenario_name.startswith("lag14"):
        return 14
    if scenario_name.startswith("lag30"):
        return 30
    return None


def infer_architecture(scenario_name: str) -> str:
    if "no_cnn" in scenario_name:
        return "no_cnn"
    return "cnn"
