from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

    HAS_TORCH_GEOMETRIC = True
except ImportError:  # pragma: no cover - optional dependency
    GATConv = GCNConv = GINConv = SAGEConv = None
    HAS_TORCH_GEOMETRIC = False


def _resolve_routing_feature(name: str, runoff_outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if name == "runoff_total":
        if "runoff_total" in runoff_outputs:
            return runoff_outputs["runoff_total"]
    if name not in runoff_outputs:
        raise KeyError(f"Runoff output '{name}' is missing for routing")
    return runoff_outputs[name]


def _apply_runoff_transform(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    transform = str(transform or "identity").lower()
    if transform in {"", "none", "identity"}:
        return tensor
    if transform == "log1p":
        return torch.log1p(torch.clamp(tensor, min=0.0))
    raise ValueError("routing_model.runoff_input_transform must be one of: identity, log1p")


def _flatten_routing_feature(tensor: torch.Tensor, *, node_index: torch.Tensor | None = None) -> torch.Tensor:
    if tensor.ndim == 3:
        batch_size, time_steps, node_count = tensor.shape
        if node_index is not None and int(node_count) != int(node_index.numel()):
            tensor = tensor.index_select(2, node_index)
            node_count = int(tensor.shape[2])
        return tensor.reshape(batch_size, time_steps, node_count, 1)
    if tensor.ndim == 4 and node_index is not None and tensor.shape[-1] == int(node_index.numel()):
        batch_size, time_steps, channels, node_count = tensor.shape
        return tensor.permute(0, 1, 3, 2).reshape(batch_size, time_steps, node_count, channels)
    if tensor.ndim == 4:
        batch_size, time_steps, y_size, x_size = tensor.shape
        flat = tensor.reshape(batch_size, time_steps, y_size * x_size, 1)
        return flat.index_select(2, node_index) if node_index is not None else flat
    if tensor.ndim == 5:
        batch_size, time_steps, channels, y_size, x_size = tensor.shape
        flat = tensor.permute(0, 1, 3, 4, 2).reshape(batch_size, time_steps, y_size * x_size, channels)
        return flat.index_select(2, node_index) if node_index is not None else flat
    raise ValueError(f"Unsupported Runoff routing feature shape {tuple(tensor.shape)}")


def _with_temporal_lags(
    tensor: torch.Tensor,
    *,
    lags: Sequence[int],
    fill_mode: str = "zero",
) -> torch.Tensor:
    if not lags:
        return tensor
    lagged_parts = []
    for lag in lags:
        lag = int(lag)
        if lag < 0:
            raise ValueError("runoff_lags must be non-negative integers")
        if lag == 0:
            lagged_parts.append(tensor)
            continue
        if lag >= int(tensor.shape[1]):
            if fill_mode == "repeat_first":
                fill = tensor[:, :1].expand(-1, tensor.shape[1], -1, -1)
            elif fill_mode == "zero":
                fill = torch.zeros_like(tensor)
            else:
                raise ValueError("runoff_lag_fill must be one of: zero, repeat_first")
            lagged_parts.append(fill)
            continue
        if fill_mode == "repeat_first":
            prefix = tensor[:, :1].expand(-1, lag, -1, -1)
        elif fill_mode == "zero":
            prefix = torch.zeros_like(tensor[:, :lag])
        else:
            raise ValueError("runoff_lag_fill must be one of: zero, repeat_first")
        lagged_parts.append(torch.cat([prefix, tensor[:, :-lag]], dim=1))
    return torch.cat(lagged_parts, dim=-1)


def _flatten_dynamic_group(tensor: torch.Tensor, *, node_index: torch.Tensor | None = None) -> torch.Tensor:
    if tensor.ndim == 4:
        batch_size, time_steps, channels, node_count = tensor.shape
        if node_index is not None and int(node_index.numel()) != int(node_count):
            raise ValueError(
                f"Compact dynamic routing tensor node count {node_count} does not match graph nodes {int(node_index.numel())}"
            )
        return tensor.permute(0, 1, 3, 2).reshape(batch_size, time_steps, node_count, channels)
    if tensor.ndim != 5:
        raise ValueError(f"Expected dynamic routing tensor [B,T,C,Y,X] or [B,T,C,N], got {tuple(tensor.shape)}")
    batch_size, time_steps, channels, y_size, x_size = tensor.shape
    flat = tensor.permute(0, 1, 3, 4, 2).reshape(batch_size, time_steps, y_size * x_size, channels)
    return flat.index_select(2, node_index) if node_index is not None else flat


def _flatten_static_group(
    tensor: torch.Tensor,
    time_steps: int,
    *,
    node_index: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor.ndim == 3:
        batch_size, channels, node_count = tensor.shape
        if node_index is not None and int(node_index.numel()) != int(node_count):
            raise ValueError(
                f"Compact static routing tensor node count {node_count} does not match graph nodes {int(node_index.numel())}"
            )
        spatial = tensor.permute(0, 2, 1).reshape(batch_size, node_count, channels)
        return spatial.unsqueeze(1).expand(-1, time_steps, -1, -1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected static routing tensor [B,C,Y,X] or [B,C,N], got {tuple(tensor.shape)}")
    batch_size, channels, y_size, x_size = tensor.shape
    spatial = tensor.permute(0, 2, 3, 1).reshape(batch_size, y_size * x_size, channels)
    if node_index is not None:
        spatial = spatial.index_select(1, node_index)
    return spatial.unsqueeze(1).expand(-1, time_steps, -1, -1)


def _resolve_gauge_index(graph: Mapping[str, Any], preferred_key: str | None = None) -> torch.Tensor | None:
    candidate_keys = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["gauge_index", "gauge_indices", "outlet_index", "outlet_indices"])
    for key in candidate_keys:
        if key in graph:
            values = torch.as_tensor(graph[key], dtype=torch.long)
            return values.reshape(-1)
    if "gauge_mask" in graph:
        mask = torch.as_tensor(graph["gauge_mask"], dtype=torch.bool)
        return torch.nonzero(mask, as_tuple=False).reshape(-1)
    return None


def _repeat_edge_index(edge_index: torch.Tensor, *, num_graphs: int, num_nodes: int) -> torch.Tensor:
    if num_graphs == 1:
        return edge_index
    offsets = torch.arange(num_graphs, dtype=edge_index.dtype, device=edge_index.device).view(num_graphs, 1, 1)
    repeated = edge_index.unsqueeze(0) + offsets * int(num_nodes)
    return repeated.permute(1, 0, 2).reshape(2, num_graphs * int(edge_index.shape[1]))


def _repeat_edge_values(values: torch.Tensor | None, *, num_graphs: int) -> torch.Tensor | None:
    if values is None or num_graphs == 1:
        return values
    if values.ndim == 1:
        return values.unsqueeze(0).expand(num_graphs, -1).reshape(-1)
    return values.unsqueeze(0).expand(num_graphs, -1, -1).reshape(num_graphs * values.shape[0], values.shape[1])


def _apply_temporal_reduction(
    prediction: torch.Tensor,
    *,
    temporal_reduction: str | None,
    steps_per_output: int | None,
) -> torch.Tensor:
    if temporal_reduction in {None, "none"}:
        return prediction

    if prediction.ndim < 2:
        raise ValueError(
            "Temporal reduction requires at least [B, T, ...] predictions, "
            f"got shape {tuple(prediction.shape)}"
        )

    if temporal_reduction in {"sum", "mean"}:
        reducer = torch.sum if temporal_reduction == "sum" else torch.mean
        return reducer(prediction, dim=1, keepdim=True)

    if temporal_reduction in {"chunk_sum", "chunk_mean", "daily_sum", "daily_mean"}:
        if steps_per_output is None or int(steps_per_output) <= 0:
            raise ValueError(
                f"Temporal reduction '{temporal_reduction}' requires a positive steps_per_output"
            )
        chunk = int(steps_per_output)
        n_steps = int(prediction.shape[1])
        if n_steps % chunk != 0:
            raise ValueError(
                f"Prediction time length {n_steps} is not divisible by steps_per_output={chunk}"
            )
        grouped = prediction.reshape(prediction.shape[0], n_steps // chunk, chunk, *prediction.shape[2:])
        reducer = torch.sum if temporal_reduction in {"chunk_sum", "daily_sum"} else torch.mean
        return reducer(grouped, dim=2)

    raise ValueError(f"Unsupported temporal_reduction '{temporal_reduction}'")


class GraphRoutingModel(nn.Module):
    """
    Flexible GNN routing model for Routing grid runoff -> gauge/node streamflow.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        runoff_output_keys: Sequence[str] = ("runoff_total",),
        static_input_keys: Sequence[str] = (),
        dynamic_input_keys: Sequence[str] = (),
        graph_key: str = "routing_graph",
        node_index_key: str = "flat_index",
        graph_node_feature_key: str | None = None,
        edge_weight_key: str | None = None,
        edge_attr_key: str | None = None,
        gauge_index_key: str | None = None,
        conv_type: str = "GCN",
        dropout: float = 0.0,
        gat_heads: int = 4,
        output_dim: int = 1,
        temporal_reduction: str | None = None,
        steps_per_output: int | None = None,
        edge_feature_dim: int | None = None,
        temporal_graph_batch_size: int | None = 32,
        runoff_lags: Sequence[int] = (),
        runoff_lag_fill: str = "zero",
        sanitize_nonfinite: bool = False,
        normalize_graph_node_features: bool = False,
        normalize_runoff_inputs: bool = False,
        runoff_input_transform: str = "identity",
        runoff_input_scaler: Mapping[str, Any] | None = None,
        temporal_head: str | None = None,
        temporal_head_kernel_size: int = 3,
        temporal_head_layers: int = 1,
        temporal_head_hidden_dim: int | None = None,
        temporal_head_residual: bool = True,
        output_activation: str = "none",
        feature_clip: float | None = None,
    ):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GraphRoutingModel")
        self.runoff_output_keys = tuple(str(key) for key in runoff_output_keys)
        self.static_input_keys = tuple(str(key) for key in static_input_keys)
        self.dynamic_input_keys = tuple(str(key) for key in dynamic_input_keys)
        self.graph_key = graph_key
        self.node_index_key = str(node_index_key)
        self.graph_node_feature_key = (
            None if graph_node_feature_key in {None, ""} else str(graph_node_feature_key)
        )
        self.edge_weight_key = None if edge_weight_key in {None, ""} else str(edge_weight_key)
        self.edge_attr_key = None if edge_attr_key in {None, ""} else str(edge_attr_key)
        self.gauge_index_key = gauge_index_key
        self.conv_type = str(conv_type).upper()
        self.dropout = float(dropout)
        self.gat_heads = int(gat_heads)
        self.output_dim = int(output_dim)
        self.temporal_reduction = temporal_reduction
        self.steps_per_output = None if steps_per_output is None else int(steps_per_output)
        self.edge_feature_dim = None if edge_feature_dim in {None, 0} else int(edge_feature_dim)
        self.temporal_graph_batch_size = None if temporal_graph_batch_size in {None, 0} else int(temporal_graph_batch_size)
        self.runoff_lags = tuple(int(lag) for lag in runoff_lags)
        self.runoff_lag_fill = str(runoff_lag_fill)
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self.normalize_graph_node_features = bool(normalize_graph_node_features)
        self.normalize_runoff_inputs = bool(normalize_runoff_inputs)
        self.runoff_input_transform = str(runoff_input_transform or "identity").lower()
        self.runoff_input_scaler = dict(runoff_input_scaler or {})
        self.temporal_head_type = str(temporal_head or "none").lower()
        self.temporal_head_residual = bool(temporal_head_residual)
        self.output_activation = str(output_activation).lower()
        self.feature_clip = None if feature_clip in {None, 0} else float(feature_clip)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = int(input_dim)
        for _ in range(int(num_layers)):
            conv, out_dim = self._make_conv(current_dim, int(hidden_dim))
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_dim))
            current_dim = out_dim
        self.head = nn.Linear(current_dim, self.output_dim)
        self.temporal_head = self._make_temporal_head(
            self.output_dim,
            head_type=self.temporal_head_type,
            kernel_size=int(temporal_head_kernel_size),
            layers=int(temporal_head_layers),
            hidden_dim=temporal_head_hidden_dim,
        )

    @staticmethod
    def _make_temporal_head(
        output_dim: int,
        *,
        head_type: str,
        kernel_size: int,
        layers: int,
        hidden_dim: int | None,
    ) -> nn.Module | None:
        if head_type in {"", "none", "identity"}:
            return None
        if head_type not in {"conv1d", "temporal_conv", "conv"}:
            raise ValueError("routing_model.temporal_head must be one of: none, conv1d")
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            raise ValueError("routing_model.temporal_head_kernel_size must be odd to preserve sequence length")
        layers = max(1, int(layers))
        hidden_channels = int(hidden_dim or output_dim)
        if hidden_channels <= 0:
            hidden_channels = int(output_dim)

        modules: list[nn.Module] = []
        in_channels = int(output_dim)
        for _ in range(max(1, layers - 1)):
            modules.append(
                nn.Conv1d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            modules.append(nn.ReLU())
            in_channels = hidden_channels
        modules.append(
            nn.Conv1d(
                in_channels,
                int(output_dim),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        return nn.Sequential(*modules)

    def _apply_output_activation(self, values: torch.Tensor) -> torch.Tensor:
        if self.output_activation in {"", "none", "identity"}:
            return values
        if self.output_activation == "softplus":
            return F.softplus(values)
        if self.output_activation == "relu":
            return F.relu(values)
        raise ValueError("routing_model.output_activation must be one of: none, softplus, relu")

    def _make_conv(self, in_dim: int, hidden_dim: int):
        if self.conv_type == "GCN":
            return GCNConv(in_dim, hidden_dim), hidden_dim
        if self.conv_type == "SAGE":
            return SAGEConv(in_dim, hidden_dim), hidden_dim
        if self.conv_type == "GAT":
            if hidden_dim % self.gat_heads != 0:
                raise ValueError("hidden_dim must be divisible by gat_heads for GAT routing")
            per_head = hidden_dim // self.gat_heads
            gat_kwargs: dict[str, Any] = {
                "heads": self.gat_heads,
                "concat": True,
                "dropout": self.dropout,
            }
            if self.edge_feature_dim is not None:
                gat_kwargs["edge_dim"] = self.edge_feature_dim
            return GATConv(in_dim, per_head, **gat_kwargs), hidden_dim
        if self.conv_type == "GIN":
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            return GINConv(mlp), hidden_dim
        raise ValueError(f"Unsupported routing conv_type '{self.conv_type}'")

    def _runoff_scaler_entry(self, key: str) -> Mapping[str, Any] | None:
        stats = self.runoff_input_scaler or {}
        if "keys" in stats and isinstance(stats["keys"], Mapping):
            return stats["keys"].get(key)
        return stats.get(key) if isinstance(stats.get(key), Mapping) else None

    def _normalize_runoff_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        entry = self._runoff_scaler_entry(key)
        transform = str(
            (entry or {}).get("transform", self.runoff_input_scaler.get("transform", self.runoff_input_transform))
        ).lower()
        tensor = _apply_runoff_transform(tensor, transform)

        if not self.normalize_runoff_inputs:
            return tensor
        if entry is None:
            raise ValueError(
                f"routing_model.normalize_runoff_inputs=true but scaler stats are missing for runoff key '{key}'. "
                "Compute train-period routing runoff stats first."
            )
        mean = torch.as_tensor(float(entry.get("mean", 0.0)), dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(float(max(float(entry.get("std", 1.0)), 1.0e-6)), dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean) / std

    def _resolve_edge_weight(self, graph: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.edge_weight_key:
            return None
        if self.edge_weight_key not in graph:
            return None
        edge_weight = torch.as_tensor(graph[self.edge_weight_key], device=device)
        if edge_weight.ndim > 1:
            edge_weight = edge_weight.reshape(-1)
        edge_weight = edge_weight.to(dtype=dtype)
        if self.sanitize_nonfinite:
            edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        return edge_weight

    def _resolve_edge_attr(self, graph: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.edge_attr_key:
            return None
        if self.edge_attr_key not in graph:
            return None
        edge_attr = torch.as_tensor(graph[self.edge_attr_key], device=device)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.ndim != 2:
            raise ValueError(
                f"routing_graph['{self.edge_attr_key}'] must have shape [E,C], got {tuple(edge_attr.shape)}"
            )
        edge_attr = edge_attr.to(dtype=dtype)
        if self.sanitize_nonfinite:
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        return edge_attr

    def _apply_conv(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.conv_type == "GCN":
            return conv(x, edge_index, edge_weight=edge_weight)
        if self.conv_type == "GAT":
            if edge_attr is not None:
                return conv(x, edge_index, edge_attr=edge_attr)
            return conv(x, edge_index)
        return conv(x, edge_index)

    def _build_node_features(
        self,
        runoff_outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        graph: Mapping[str, Any],
        *,
        node_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_parts = []
        time_steps = None
        for key in self.runoff_output_keys:
            tensor = _flatten_routing_feature(_resolve_routing_feature(key, runoff_outputs), node_index=node_index)
            tensor = self._normalize_runoff_tensor(key, tensor)
            tensor = _with_temporal_lags(tensor, lags=self.runoff_lags, fill_mode=self.runoff_lag_fill)
            time_steps = tensor.shape[1]
            feature_parts.append(tensor)
        for key in self.dynamic_input_keys:
            if key not in batch:
                raise KeyError(f"Batch is missing routing dynamic input '{key}'")
            tensor = _flatten_dynamic_group(batch[key], node_index=node_index)
            time_steps = tensor.shape[1]
            feature_parts.append(tensor)
        if time_steps is None:
            raise ValueError("GraphRoutingModel requires at least one temporal feature input")
        for key in self.static_input_keys:
            if key not in batch:
                raise KeyError(f"Batch is missing routing static input '{key}'")
            feature_parts.append(_flatten_static_group(batch[key], time_steps, node_index=node_index))
        if self.graph_node_feature_key:
            if self.graph_node_feature_key not in graph:
                raise KeyError(
                    f"routing_graph is missing graph node feature '{self.graph_node_feature_key}'"
                )
            node_features = torch.as_tensor(
                graph[self.graph_node_feature_key],
                dtype=feature_parts[0].dtype,
                device=feature_parts[0].device,
            )
            if node_features.ndim == 1:
                node_features = node_features.unsqueeze(-1)
            if node_features.ndim != 2:
                raise ValueError(
                    f"routing_graph['{self.graph_node_feature_key}'] must have shape [N,C], "
                    f"got {tuple(node_features.shape)}"
                )
            if self.sanitize_nonfinite:
                node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            if self.normalize_graph_node_features:
                mean = node_features.mean(dim=0, keepdim=True)
                std = node_features.std(dim=0, keepdim=True).clamp_min(torch.finfo(node_features.dtype).eps)
                node_features = (node_features - mean) / std
            feature_parts.append(
                node_features.unsqueeze(0).unsqueeze(0).expand(
                    feature_parts[0].shape[0],
                    time_steps,
                    -1,
                    -1,
                )
            )
        features = torch.cat(feature_parts, dim=-1)
        if self.sanitize_nonfinite:
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_clip is not None:
            features = torch.clamp(features, min=-self.feature_clip, max=self.feature_clip)
        return features

    def _route_feature_chunk(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        gauge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, time_steps, node_count, feature_dim = features.shape
        graph_count = batch_size * time_steps
        x = features.reshape(graph_count * node_count, feature_dim)
        chunk_edge_index = _repeat_edge_index(edge_index, num_graphs=graph_count, num_nodes=node_count)
        chunk_edge_weight = _repeat_edge_values(edge_weight, num_graphs=graph_count)
        chunk_edge_attr = _repeat_edge_values(edge_attr, num_graphs=graph_count)

        for conv, norm in zip(self.convs, self.norms):
            x = self._apply_conv(conv, x, chunk_edge_index, edge_weight=chunk_edge_weight, edge_attr=chunk_edge_attr)
            x = norm(x)
            x = F.relu(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        y_hat = self._apply_output_activation(self.head(x)).reshape(
            batch_size,
            time_steps,
            node_count,
            self.output_dim,
        )
        if gauge_index is not None:
            y_hat = y_hat.index_select(2, gauge_index)
        return y_hat

    def _apply_temporal_head(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.temporal_head is None:
            return prediction
        batch_size, time_steps, node_count, output_dim = prediction.shape
        x = prediction.permute(0, 2, 3, 1).reshape(batch_size * node_count, output_dim, time_steps)
        y = self.temporal_head(x)
        y = y.reshape(batch_size, node_count, output_dim, time_steps).permute(0, 3, 1, 2)
        if self.temporal_head_residual:
            y = y + prediction
        return y

    def forward(self, runoff_outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> torch.Tensor:
        if self.graph_key not in batch:
            raise KeyError(f"Batch is missing graph payload '{self.graph_key}'")
        graph = batch[self.graph_key]
        if not isinstance(graph, Mapping) or "edge_index" not in graph:
            raise ValueError("routing_graph must be a mapping containing 'edge_index'")
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=next(self.parameters()).device)
        gauge_index = _resolve_gauge_index(graph, preferred_key=self.gauge_index_key)
        if gauge_index is not None:
            gauge_index = gauge_index.to(device=edge_index.device)
        node_index = graph.get(self.node_index_key)
        if node_index is not None:
            node_index = torch.as_tensor(node_index, dtype=torch.long, device=edge_index.device).reshape(-1)

        features = self._build_node_features(runoff_outputs, batch, graph, node_index=node_index)
        edge_weight = self._resolve_edge_weight(graph, device=edge_index.device, dtype=features.dtype)
        edge_attr = self._resolve_edge_attr(graph, device=edge_index.device, dtype=features.dtype)
        batch_size, time_steps, node_count, _ = features.shape
        if "num_nodes" in graph and int(graph["num_nodes"]) != int(node_count):
            raise ValueError(
                f"routing_graph num_nodes={int(graph['num_nodes'])} does not match feature node count {int(node_count)}"
            )

        chunk_size = time_steps if self.temporal_graph_batch_size is None else max(1, int(self.temporal_graph_batch_size))
        routed_chunks = []
        for start in range(0, time_steps, chunk_size):
            chunk = features[:, start : start + chunk_size].contiguous()
            routed_chunks.append(
                self._route_feature_chunk(
                    chunk,
                    edge_index,
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                    gauge_index=gauge_index,
                )
            )

        prediction = torch.cat(routed_chunks, dim=1)
        prediction = self._apply_temporal_head(prediction)
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        return _apply_temporal_reduction(
            prediction,
            temporal_reduction=self.temporal_reduction,
            steps_per_output=self.steps_per_output,
        )
