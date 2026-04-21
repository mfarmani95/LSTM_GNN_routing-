from __future__ import annotations

from typing import Any, Mapping

import torch

from lstm_gnn_routing.routing_models.gnn_routing import GraphRoutingModel, HAS_TORCH_GEOMETRIC
from lstm_gnn_routing.routing_models.runoff_transfer import GridToGraphRunoffTransfer
from lstm_gnn_routing.runoff_models.precomputed_runoff import PrecomputedRunoffModel
from lstm_gnn_routing.runoff_models.lstm_runoff import (
    SpatialLSTMRunoffModel,
    SpatialTemporalConvRunoffModel,
)
from lstm_gnn_routing.utils.data import load_scaler_yaml


def _runoff_transfer_config(config) -> dict[str, Any]:
    transfer_cfg = config.section("runoff_transfer")
    if transfer_cfg:
        return transfer_cfg
    routing_cfg = config.section("routing")
    nested = routing_cfg.get("runoff_transfer", {}) if isinstance(routing_cfg, Mapping) else {}
    return dict(nested or {})


def _dynamic_input_dim(tensor: torch.Tensor) -> int:
    if tensor.ndim in {4, 5}:
        return int(tensor.shape[2])
    raise ValueError(f"Dynamic runoff input must be [B,T,C,N] or [B,T,C,Y,X], got {tuple(tensor.shape)}")


def _static_input_dim(tensor: torch.Tensor) -> int:
    if tensor.ndim in {3, 4}:
        return int(tensor.shape[1])
    raise ValueError(f"Static runoff input must be [B,C,N] or [B,C,Y,X], got {tuple(tensor.shape)}")


def build_runoff_model(config, *, example_batch: Mapping[str, Any], device=None):
    runoff_cfg = config.section("runoff_model")
    model_type = str(runoff_cfg.get("type", "lstm")).lower()
    if model_type not in {"lstm", "spatial_lstm", "temporal_conv", "tcn", "spatial_temporal_conv", "precomputed", "identity", "passthrough"}:
        raise ValueError(f"Unsupported runoff_model.type '{model_type}'")

    dynamic_input_keys = tuple(runoff_cfg.get("dynamic_input_keys", ["x_forcing_ml"]))
    static_input_keys = tuple(runoff_cfg.get("static_input_keys", []))
    if model_type in {"precomputed", "identity", "passthrough"}:
        channel_names = tuple(
            runoff_cfg.get(
                "input_channel_names",
                example_batch.get("x_info", [{}])[0].get(f"{dynamic_input_keys[0]}_names", []),
            )
        )
        model = PrecomputedRunoffModel(
            dynamic_input_keys=dynamic_input_keys,
            output_keys=tuple(runoff_cfg.get("output_keys", ["RUNSF", "RUNSB"])),
            input_channel_names=channel_names,
            sanitize_nonfinite=bool(runoff_cfg.get("sanitize_nonfinite", True)),
        )
        return model.to(device=device) if device is not None else model

    input_dim = runoff_cfg.get("input_dim")
    if input_dim is None:
        inferred_dim = 0
        for key in dynamic_input_keys:
            if key not in example_batch:
                raise KeyError(f"Example batch is missing runoff dynamic input '{key}'")
            inferred_dim += _dynamic_input_dim(example_batch[key])
        for key in static_input_keys:
            if key not in example_batch:
                raise KeyError(f"Example batch is missing runoff static input '{key}'")
            inferred_dim += _static_input_dim(example_batch[key])
        input_dim = inferred_dim

    common_kwargs = {
        "dynamic_input_keys": dynamic_input_keys,
        "static_input_keys": static_input_keys,
        "input_dim": int(input_dim),
        "hidden_dim": int(runoff_cfg.get("hidden_dim", 64)),
        "num_layers": int(runoff_cfg.get("num_layers", 1)),
        "dropout": float(runoff_cfg.get("dropout", 0.0)),
        "output_keys": tuple(runoff_cfg.get("output_keys", ["runoff_total"])),
        "output_activation": str(runoff_cfg.get("output_activation", "softplus")),
        "node_batch_size": runoff_cfg.get("node_batch_size"),
        "sanitize_nonfinite": bool(runoff_cfg.get("sanitize_nonfinite", True)),
        "feature_clip": runoff_cfg.get("feature_clip", 10.0),
    }
    if model_type in {"temporal_conv", "tcn", "spatial_temporal_conv"}:
        model = SpatialTemporalConvRunoffModel(
            **common_kwargs,
            kernel_size=int(runoff_cfg.get("kernel_size", 3)),
        )
    else:
        model = SpatialLSTMRunoffModel(
            **common_kwargs,
            input_norm=runoff_cfg.get("input_norm", "none"),
            use_cudnn=bool(runoff_cfg.get("use_cudnn", True)),
        )
    return model.to(device=device) if device is not None else model


def build_runoff_transfer_model(config, *, example_batch: Mapping[str, Any] | None = None, device=None):
    transfer_cfg = _runoff_transfer_config(config)
    if not transfer_cfg or not bool(transfer_cfg.get("enabled", True)):
        return None

    graph_key = str(transfer_cfg.get("graph_key", "routing_graph"))
    source_feature_key = str(transfer_cfg.get("source_feature_key", "runoff_source_features"))
    source_feature_dim = transfer_cfg.get("source_feature_dim")
    source_count = transfer_cfg.get("source_count")
    if example_batch is not None and graph_key in example_batch and isinstance(example_batch[graph_key], Mapping):
        graph = example_batch[graph_key]
        if source_feature_dim in {None, 0} and source_feature_key in graph:
            features = torch.as_tensor(graph[source_feature_key])
            source_feature_dim = 1 if features.ndim == 1 else int(features.shape[-1])
        if source_count in {None, 0}:
            target_key = str(transfer_cfg.get("target_index_key", "runoff_target_index"))
            if target_key in graph:
                source_count = int(torch.as_tensor(graph[target_key]).numel())

    model = GridToGraphRunoffTransfer(
        mode=str(transfer_cfg.get("type", transfer_cfg.get("mode", "fixed"))).lower(),
        graph_key=graph_key,
        output_keys=tuple(transfer_cfg.get("output_keys", [])),
        source_index_key=str(transfer_cfg.get("source_index_key", "runoff_source_index")),
        source_flat_index_key=str(transfer_cfg.get("source_flat_index_key", "runoff_source_flat_index")),
        target_index_key=str(transfer_cfg.get("target_index_key", "runoff_target_index")),
        weight_key=str(transfer_cfg.get("weight_key", "runoff_source_weight")),
        source_feature_key=source_feature_key,
        source_feature_dim=None if source_feature_dim in {None, 0} else int(source_feature_dim),
        source_count=None if source_count in {None, 0} else int(source_count),
        hidden_dim=int(transfer_cfg.get("hidden_dim", 16)),
        normalize_by_target=bool(transfer_cfg.get("normalize_by_target", False)),
        weight_activation=str(transfer_cfg.get("weight_activation", "sigmoid_scale")),
        sanitize_nonfinite=bool(transfer_cfg.get("sanitize_nonfinite", True)),
    )
    return model.to(device=device) if device is not None else model


def _routing_runoff_scaler(config, routing_cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not bool(routing_cfg.get("normalize_runoff_inputs", False)):
        return None
    scaler_cfg = config.section("scaler")
    scaler_path = scaler_cfg.get("path", scaler_cfg.get("file_path"))
    if scaler_path in (None, ""):
        raise ValueError("routing_model.normalize_runoff_inputs=true requires scaler.path.")
    scaler = load_scaler_yaml(scaler_path)
    group_name = str(routing_cfg.get("runoff_scaler_group", "routing_runoff"))
    stats = (scaler.get("routing_runoff_stats") or {}).get(group_name)
    if not stats:
        raise ValueError(f"Scaler file {scaler_path} is missing routing_runoff_stats.{group_name}.")
    return stats


def _routing_input_dim(config, example_batch: Mapping[str, Any]) -> int:
    routing_cfg = config.section("routing_model")
    input_dim = 0
    runoff_lag_count = len(tuple(routing_cfg.get("runoff_lags", []))) or 1
    for _ in routing_cfg.get("runoff_output_keys", ["runoff_total"]):
        input_dim += runoff_lag_count
    for key in routing_cfg.get("dynamic_input_keys", []):
        if key not in example_batch:
            raise KeyError(f"Example batch is missing routing dynamic input '{key}'")
        input_dim += _dynamic_input_dim(example_batch[key])
    for key in routing_cfg.get("static_input_keys", []):
        if key not in example_batch:
            raise KeyError(f"Example batch is missing routing static input '{key}'")
        input_dim += _static_input_dim(example_batch[key])
    graph_node_feature_key = routing_cfg.get("graph_node_feature_key")
    if graph_node_feature_key:
        graph = example_batch.get(str(routing_cfg.get("graph_key", "routing_graph")))
        if not isinstance(graph, Mapping):
            raise KeyError("Example batch is missing routing graph payload required for graph node features")
        node_features = torch.as_tensor(graph[graph_node_feature_key])
        input_dim += 1 if node_features.ndim == 1 else int(node_features.shape[1])
    return input_dim


def _routing_edge_feature_dim(config, example_batch: Mapping[str, Any] | None) -> int | None:
    if example_batch is None:
        return None
    routing_cfg = config.section("routing_model")
    edge_attr_key = routing_cfg.get("edge_attr_key")
    if not edge_attr_key:
        return None
    graph = example_batch.get(str(routing_cfg.get("graph_key", "routing_graph")))
    if not isinstance(graph, Mapping) or edge_attr_key not in graph:
        return None
    edge_attr = torch.as_tensor(graph[edge_attr_key])
    return 1 if edge_attr.ndim == 1 else int(edge_attr.shape[1])


def build_routing_model(config, *, example_batch: Mapping[str, Any], device=None):
    routing_cfg = config.section("routing_model")
    if str(routing_cfg.get("type", "gnn")).lower() != "gnn":
        raise ValueError("This standalone trainer currently supports routing_model.type='gnn'.")
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("routing_model.type='gnn' requires torch_geometric")
    input_dim = routing_cfg.get("input_dim")
    if input_dim is None:
        input_dim = _routing_input_dim(config, example_batch)

    model = GraphRoutingModel(
        input_dim=int(input_dim),
        hidden_dim=int(routing_cfg.get("hidden_dim", 64)),
        num_layers=int(routing_cfg.get("num_layers", 2)),
        runoff_output_keys=tuple(routing_cfg.get("runoff_output_keys", ["runoff_total"])),
        static_input_keys=tuple(routing_cfg.get("static_input_keys", [])),
        dynamic_input_keys=tuple(routing_cfg.get("dynamic_input_keys", [])),
        graph_key=str(routing_cfg.get("graph_key", "routing_graph")),
        node_index_key=str(routing_cfg.get("node_index_key", "flat_index")),
        graph_node_feature_key=routing_cfg.get("graph_node_feature_key"),
        edge_weight_key=routing_cfg.get("edge_weight_key", "edge_weight"),
        edge_attr_key=routing_cfg.get("edge_attr_key"),
        gauge_index_key=routing_cfg.get("gauge_index_key"),
        conv_type=str(routing_cfg.get("conv_type", "GCN")),
        dropout=float(routing_cfg.get("dropout", 0.0)),
        gat_heads=int(routing_cfg.get("gat_heads", 4)),
        output_dim=int(routing_cfg.get("output_dim", 1)),
        temporal_reduction=routing_cfg.get("temporal_reduction"),
        steps_per_output=routing_cfg.get("steps_per_output"),
        edge_feature_dim=_routing_edge_feature_dim(config, example_batch),
        temporal_graph_batch_size=routing_cfg.get("temporal_graph_batch_size", 32),
        runoff_lags=tuple(routing_cfg.get("runoff_lags", [])),
        runoff_lag_fill=str(routing_cfg.get("runoff_lag_fill", "zero")),
        sanitize_nonfinite=bool(routing_cfg.get("sanitize_nonfinite", True)),
        normalize_graph_node_features=bool(routing_cfg.get("normalize_graph_node_features", False)),
        normalize_runoff_inputs=bool(routing_cfg.get("normalize_runoff_inputs", False)),
        runoff_input_transform=str(routing_cfg.get("runoff_input_transform", "identity")),
        runoff_input_scaler=_routing_runoff_scaler(config, routing_cfg),
        temporal_head=routing_cfg.get("temporal_head"),
        temporal_head_kernel_size=int(routing_cfg.get("temporal_head_kernel_size", 3)),
        temporal_head_layers=int(routing_cfg.get("temporal_head_layers", 1)),
        temporal_head_hidden_dim=routing_cfg.get("temporal_head_hidden_dim"),
        temporal_head_residual=bool(routing_cfg.get("temporal_head_residual", True)),
        output_activation=str(routing_cfg.get("output_activation", "none")),
        feature_clip=routing_cfg.get("feature_clip", 10.0),
    )
    return model.to(device=device) if device is not None else model
