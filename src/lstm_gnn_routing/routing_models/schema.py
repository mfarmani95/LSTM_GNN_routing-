from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _to_long_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long)


def _to_float_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _resolve_alias(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _infer_num_nodes(graph: Mapping[str, Any]) -> int:
    candidates: list[int] = []
    if "num_nodes" in graph:
        candidates.append(int(graph["num_nodes"]))
    if "edge_index" in graph:
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        if edge_index.numel() > 0:
            candidates.append(int(edge_index.max().item()) + 1)
    if "flat_index" in graph:
        candidates.append(int(torch.as_tensor(graph["flat_index"]).numel()))
    if "node_y" in graph and "node_x" in graph:
        candidates.append(int(torch.as_tensor(graph["node_y"]).numel()))
    if "node_features" in graph:
        candidates.append(int(torch.as_tensor(graph["node_features"]).shape[0]))
    if "gauge_mask" in graph:
        candidates.append(int(torch.as_tensor(graph["gauge_mask"]).numel()))
    if "routing_matrix" in graph:
        matrix = torch.as_tensor(graph["routing_matrix"])
        if matrix.ndim == 2:
            candidates.append(int(matrix.shape[-1]))
    if "runoff_target_index" in graph:
        target_index = torch.as_tensor(graph["runoff_target_index"], dtype=torch.long)
        if target_index.numel() > 0:
            candidates.append(int(target_index.max().item()) + 1)
    if not candidates:
        raise ValueError("Could not infer routing graph num_nodes from payload")
    return max(candidates)


def _flatten_index_from_grid(
    node_y: torch.Tensor,
    node_x: torch.Tensor,
    *,
    grid_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    if grid_shape is None:
        raise ValueError("grid_shape is required to build flat_index from node_y/node_x")
    y_size, x_size = int(grid_shape[0]), int(grid_shape[1])
    return node_y * x_size + node_x


def normalize_routing_graph_payload(
    payload: Any,
    *,
    dtype: torch.dtype = torch.float32,
    grid_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Routing graph payload must be a mapping/dict-like object")
    raw = dict(payload)
    graph: dict[str, Any] = {}

    edge_index = _resolve_alias(raw, "edge_index", "edges")
    if edge_index is None:
        raise KeyError("Routing graph must define 'edge_index' or 'edges'")
    edge_index = _to_long_tensor(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    graph["edge_index"] = edge_index

    edge_attr = _resolve_alias(raw, "edge_attr", "edge_attributes", "edge_features")
    if edge_attr is not None:
        edge_attr = _to_float_tensor(edge_attr, dtype)
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_attr first dimension must match number of edges {edge_index.shape[1]}, got {tuple(edge_attr.shape)}"
            )
        graph["edge_attr"] = edge_attr
        graph["edge_feature_names"] = _normalize_list(_resolve_alias(raw, "edge_feature_names", "edge_attr_names"))

    edge_weight = _resolve_alias(raw, "edge_weight", "edge_weights")
    if edge_weight is not None:
        edge_weight = _to_float_tensor(edge_weight, dtype).reshape(-1)
        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weight length must match number of edges {edge_index.shape[1]}, got {tuple(edge_weight.shape)}"
            )
        graph["edge_weight"] = edge_weight

    node_features = _resolve_alias(raw, "node_features", "x", "node_attr")
    if node_features is not None:
        graph["node_features"] = _to_float_tensor(node_features, dtype)

    node_y = _resolve_alias(raw, "node_y", "y_index", "row_index")
    node_x = _resolve_alias(raw, "node_x", "x_index", "col_index")
    flat_index = _resolve_alias(raw, "flat_index", "node_to_grid_index", "grid_index")
    if flat_index is not None:
        graph["flat_index"] = _to_long_tensor(flat_index).reshape(-1)
    elif node_y is not None and node_x is not None:
        node_y_tensor = _to_long_tensor(node_y).reshape(-1)
        node_x_tensor = _to_long_tensor(node_x).reshape(-1)
        if node_y_tensor.shape != node_x_tensor.shape:
            raise ValueError("node_y and node_x must have the same shape")
        graph["node_y"] = node_y_tensor
        graph["node_x"] = node_x_tensor
        graph["flat_index"] = _flatten_index_from_grid(node_y_tensor, node_x_tensor, grid_shape=grid_shape)

    gauge_index = _resolve_alias(raw, "gauge_index", "gauge_indices", "outlet_index", "outlet_indices")
    gauge_mask = _resolve_alias(raw, "gauge_mask", "outlet_mask")
    if gauge_index is not None:
        graph["gauge_index"] = _to_long_tensor(gauge_index).reshape(-1)
    elif gauge_mask is not None:
        mask = torch.as_tensor(gauge_mask, dtype=torch.bool).reshape(-1)
        graph["gauge_mask"] = mask
        graph["gauge_index"] = torch.nonzero(mask, as_tuple=False).reshape(-1)

    routing_matrix = _resolve_alias(raw, "routing_matrix", "adjacency_matrix", "transport_matrix")
    if routing_matrix is not None:
        routing_matrix = _to_float_tensor(routing_matrix, dtype)
        if routing_matrix.ndim != 2:
            raise ValueError(f"routing_matrix must be rank-2, got {tuple(routing_matrix.shape)}")
        graph["routing_matrix"] = routing_matrix

    runoff_target_index = _resolve_alias(raw, "runoff_target_index", "source_to_node", "grid_to_node_index")
    if runoff_target_index is not None:
        graph["runoff_target_index"] = _to_long_tensor(runoff_target_index).reshape(-1)

    runoff_source_index = _resolve_alias(raw, "runoff_source_index", "source_index")
    if runoff_source_index is not None:
        graph["runoff_source_index"] = _to_long_tensor(runoff_source_index).reshape(-1)

    runoff_source_flat_index = _resolve_alias(
        raw,
        "runoff_source_flat_index",
        "source_flat_index",
        "runoff_grid_flat_index",
    )
    if runoff_source_flat_index is not None:
        graph["runoff_source_flat_index"] = _to_long_tensor(runoff_source_flat_index).reshape(-1)

    runoff_source_weight = _resolve_alias(raw, "runoff_source_weight", "source_weight", "grid_to_node_weight")
    if runoff_source_weight is not None:
        graph["runoff_source_weight"] = _to_float_tensor(runoff_source_weight, dtype).reshape(-1)

    runoff_source_features = _resolve_alias(
        raw,
        "runoff_source_features",
        "source_features",
        "grid_to_node_features",
    )
    if runoff_source_features is not None:
        source_features = _to_float_tensor(runoff_source_features, dtype)
        if source_features.ndim == 1:
            source_features = source_features.unsqueeze(-1)
        graph["runoff_source_features"] = source_features
        graph["runoff_source_feature_names"] = _normalize_list(
            _resolve_alias(raw, "runoff_source_feature_names", "source_feature_names")
        )

    num_nodes = _infer_num_nodes({**raw, **graph})
    graph["num_nodes"] = int(num_nodes)
    graph["num_edges"] = int(edge_index.shape[1])

    if "flat_index" in graph and graph["flat_index"].numel() != num_nodes:
        raise ValueError(
            f"flat_index length {graph['flat_index'].numel()} must equal num_nodes {num_nodes}"
        )
    if "node_features" in graph and graph["node_features"].shape[0] != num_nodes:
        raise ValueError(
            f"node_features first dimension {graph['node_features'].shape[0]} must equal num_nodes {num_nodes}"
        )
    if "gauge_index" in graph:
        gauge_index = graph["gauge_index"]
        if gauge_index.numel() and int(gauge_index.max().item()) >= num_nodes:
            raise ValueError("gauge_index contains node ids outside the routing graph")
        graph["num_gauges"] = int(gauge_index.numel())
    else:
        graph["num_gauges"] = 0

    graph["node_ids"] = _normalize_list(_resolve_alias(raw, "node_ids", "node_names"))
    graph["gauge_ids"] = _normalize_list(_resolve_alias(raw, "gauge_ids", "outlet_ids", "gauge_names"))
    if "runoff_target_index" in graph:
        source_count = int(graph["runoff_target_index"].numel())
        if source_count and int(graph["runoff_target_index"].max().item()) >= num_nodes:
            raise ValueError("runoff_target_index contains node ids outside the routing graph")
        for key in ("runoff_source_index", "runoff_source_flat_index", "runoff_source_weight"):
            if key in graph and int(graph[key].numel()) != source_count:
                raise ValueError(f"{key} length must match runoff_target_index length {source_count}")
        if "runoff_source_features" in graph and int(graph["runoff_source_features"].shape[0]) != source_count:
            raise ValueError(
                "runoff_source_features first dimension must match "
                f"runoff_target_index length {source_count}"
            )
    if grid_shape is not None:
        graph["grid_shape"] = tuple(int(v) for v in grid_shape)

    metadata = dict(raw.get("metadata", {}) or {})
    metadata.update(
        {
            "num_nodes": graph["num_nodes"],
            "num_edges": graph["num_edges"],
            "num_gauges": graph["num_gauges"],
        }
    )
    if "grid_shape" in graph:
        metadata["grid_shape"] = graph["grid_shape"]
    graph["metadata"] = metadata
    return graph
