from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_long_vector(value: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)


def _as_float_vector(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)


def _resolve_graph_tensor(graph: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key and key in graph:
            return graph[key]
    return None


def _target_weight_normalize(
    weights: torch.Tensor,
    target_index: torch.Tensor,
    *,
    num_targets: int,
) -> torch.Tensor:
    denom = torch.zeros(num_targets, dtype=weights.dtype, device=weights.device)
    denom.scatter_add_(0, target_index, weights)
    return weights / denom.index_select(0, target_index).clamp_min(torch.finfo(weights.dtype).eps)


class GridToGraphRunoffTransfer(nn.Module):
    """Map grid-cell runoff onto routing-graph nodes before channel routing.

    The graph stores a sparse source-to-target mapping:
    ``source_index -> target_index`` with optional weights/features. This lets
    an ML runoff model run on active grid cells while the routing model operates
    on Ngen divide/flowpath nodes.
    """

    def __init__(
        self,
        *,
        mode: str = "fixed",
        graph_key: str = "routing_graph",
        output_keys: Sequence[str] | None = None,
        source_index_key: str = "runoff_source_index",
        source_flat_index_key: str = "runoff_source_flat_index",
        target_index_key: str = "runoff_target_index",
        weight_key: str = "runoff_source_weight",
        source_feature_key: str = "runoff_source_features",
        source_feature_dim: int | None = None,
        source_count: int | None = None,
        hidden_dim: int = 16,
        normalize_by_target: bool = False,
        weight_activation: str = "sigmoid_scale",
        sanitize_nonfinite: bool = True,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"fixed", "neural", "learned"}:
            raise ValueError("runoff_transfer.mode must be one of: fixed, neural")
        self.graph_key = str(graph_key)
        self.output_keys = tuple(str(key) for key in output_keys or ())
        self.source_index_key = str(source_index_key)
        self.source_flat_index_key = str(source_flat_index_key)
        self.target_index_key = str(target_index_key)
        self.weight_key = str(weight_key)
        self.source_feature_key = str(source_feature_key)
        self.normalize_by_target = bool(normalize_by_target)
        self.weight_activation = str(weight_activation).lower()
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self.input_keys = (self.graph_key,)

        feature_dim = None if source_feature_dim in {None, 0} else int(source_feature_dim)
        self.feature_mlp: nn.Module | None = None
        self.source_logits: nn.Parameter | None = None
        if self.mode in {"neural", "learned"}:
            if feature_dim is not None:
                hidden = max(1, int(hidden_dim))
                self.feature_mlp = nn.Sequential(
                    nn.Linear(feature_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                # Start close to the fixed transfer. The network can then learn
                # source multipliers without shocking early training.
                nn.init.zeros_(self.feature_mlp[-1].weight)
                nn.init.zeros_(self.feature_mlp[-1].bias)
            else:
                if source_count in {None, 0}:
                    raise ValueError(
                        "runoff_transfer.type='neural' requires runoff_source_features "
                        "or a known source_count for per-source weights"
                    )
                self.source_logits = nn.Parameter(torch.zeros(int(source_count)))

    def _resolve_mapping(
        self,
        graph: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, int]:
        target_raw = _resolve_graph_tensor(graph, (self.target_index_key, "runoff_target_index", "source_to_node"))
        if target_raw is None:
            raise KeyError(
                f"routing_graph is missing '{self.target_index_key}'. "
                "A grid-to-graph runoff transfer needs a source-to-target mapping."
            )
        target_index = _as_long_vector(target_raw, device=device)
        if target_index.numel() == 0:
            raise ValueError("runoff transfer mapping has zero source cells")
        if int(target_index.min().item()) < 0:
            raise ValueError("runoff transfer target indices must be non-negative")

        source_raw = _resolve_graph_tensor(
            graph,
            (self.source_index_key, "runoff_source_index", self.source_flat_index_key, "runoff_source_flat_index"),
        )
        source_index = None if source_raw is None else _as_long_vector(source_raw, device=device)
        if source_index is not None and source_index.numel() != target_index.numel():
            raise ValueError(
                f"runoff transfer source_index length {source_index.numel()} does not match "
                f"target_index length {target_index.numel()}"
            )

        weight_raw = _resolve_graph_tensor(graph, (self.weight_key, "runoff_source_weight", "source_weight"))
        if weight_raw is None:
            weights = torch.ones(target_index.numel(), dtype=dtype, device=device)
        else:
            weights = _as_float_vector(weight_raw, device=device, dtype=dtype)
            if weights.numel() != target_index.numel():
                raise ValueError(
                    f"runoff transfer weights length {weights.numel()} does not match "
                    f"target_index length {target_index.numel()}"
                )

        if "num_nodes" in graph:
            num_targets = int(graph["num_nodes"])
        else:
            num_targets = int(target_index.max().item()) + 1
        return source_index, target_index, weights, num_targets

    def _source_multipliers(
        self,
        graph: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        source_count: int,
    ) -> torch.Tensor:
        if self.mode == "fixed":
            return torch.ones(source_count, dtype=dtype, device=device)

        if self.feature_mlp is not None:
            raw_features = _resolve_graph_tensor(graph, (self.source_feature_key, "runoff_source_features"))
            if raw_features is None:
                raise KeyError(
                    f"runoff_transfer.type='neural' was built with feature weights, "
                    f"but routing_graph is missing '{self.source_feature_key}'"
                )
            features = torch.as_tensor(raw_features, dtype=dtype, device=device)
            if features.ndim == 1:
                features = features.unsqueeze(-1)
            if features.shape[0] != source_count:
                raise ValueError(
                    f"runoff source feature rows {features.shape[0]} do not match source_count {source_count}"
                )
            logits = self.feature_mlp(features).reshape(-1)
        else:
            if self.source_logits is None:
                raise RuntimeError("Neural runoff transfer was not initialized with source weights")
            if self.source_logits.numel() != source_count:
                raise ValueError(
                    f"Per-source runoff transfer weights expect {self.source_logits.numel()} sources, "
                    f"but graph mapping has {source_count}"
                )
            logits = self.source_logits.to(device=device, dtype=dtype)

        if self.weight_activation in {"", "none", "identity"}:
            multiplier = logits
        elif self.weight_activation in {"sigmoid_scale", "sigmoid"}:
            multiplier = 2.0 * torch.sigmoid(logits)
        elif self.weight_activation == "softplus":
            multiplier = F.softplus(logits) / F.softplus(torch.zeros((), dtype=dtype, device=device))
        else:
            raise ValueError(
                "runoff_transfer.weight_activation must be one of: sigmoid_scale, softplus, identity"
            )
        return multiplier

    def _source_aligned_runoff(
        self,
        tensor: torch.Tensor,
        source_index: torch.Tensor | None,
        *,
        source_count: int,
    ) -> tuple[torch.Tensor, bool]:
        if tensor.ndim == 3:
            values = tensor
            had_channel = False
        elif tensor.ndim == 4:
            # Support compact [B,T,C,N] and gridded [B,T,Y,X]. If the third
            # dimension looks like a channel axis, treat it as C.
            looks_channel_first = int(tensor.shape[2]) <= 32
            if looks_channel_first:
                values = tensor
                had_channel = True
            else:
                batch_size, time_steps, y_size, x_size = tensor.shape
                values = tensor.reshape(batch_size, time_steps, y_size * x_size)
                had_channel = False
        elif tensor.ndim == 5:
            batch_size, time_steps, channels, y_size, x_size = tensor.shape
            values = tensor.reshape(batch_size, time_steps, channels, y_size * x_size)
            had_channel = True
        else:
            raise ValueError(
                f"Runoff transfer expects [B,T,N], [B,T,Y,X], [B,T,C,N], or [B,T,C,Y,X], "
                f"got {tuple(tensor.shape)}"
            )

        node_dim = -1
        node_count = int(values.shape[node_dim])
        if node_count == source_count:
            return values, had_channel
        if source_index is not None and source_index.numel() and int(source_index.max().item()) < node_count:
            values = values.index_select(node_dim, source_index)
        else:
            raise ValueError(
                f"Runoff transfer received {node_count} source nodes, "
                f"but mapping has {source_count}; provide '{self.source_index_key}'"
            )
        return values, had_channel

    def _transfer_tensor(
        self,
        tensor: torch.Tensor,
        graph: Mapping[str, Any],
    ) -> torch.Tensor:
        source_index, target_index, base_weights, num_targets = self._resolve_mapping(
            graph,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        source_count = int(target_index.numel())
        weights = base_weights * self._source_multipliers(
            graph,
            device=tensor.device,
            dtype=tensor.dtype,
            source_count=source_count,
        )
        if self.sanitize_nonfinite:
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize_by_target:
            weights = _target_weight_normalize(weights, target_index, num_targets=num_targets)

        values, had_channel = self._source_aligned_runoff(
            tensor,
            source_index,
            source_count=source_count,
        )
        if self.sanitize_nonfinite:
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if values.ndim == 3:
            weighted = values * weights.view(1, 1, source_count)
            result = torch.zeros(
                values.shape[0],
                values.shape[1],
                num_targets,
                dtype=values.dtype,
                device=values.device,
            )
            result.scatter_add_(2, target_index.view(1, 1, source_count).expand_as(weighted), weighted)
            return result

        if values.ndim == 4:
            weighted = values * weights.view(1, 1, 1, source_count)
            result = torch.zeros(
                values.shape[0],
                values.shape[1],
                values.shape[2],
                num_targets,
                dtype=values.dtype,
                device=values.device,
            )
            result.scatter_add_(
                3,
                target_index.view(1, 1, 1, source_count).expand_as(weighted),
                weighted,
            )
            return result if had_channel else result.squeeze(2)

        raise RuntimeError(f"Unexpected source-aligned runoff shape {tuple(values.shape)}")

    def forward(self, runoff_outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.graph_key not in batch:
            raise KeyError(f"Batch is missing graph payload '{self.graph_key}' for runoff transfer")
        graph = batch[self.graph_key]
        if not isinstance(graph, Mapping):
            raise TypeError(f"Batch['{self.graph_key}'] must be a routing graph mapping")

        selected = self.output_keys or tuple(key for key, value in runoff_outputs.items() if torch.is_tensor(value))
        transferred: dict[str, torch.Tensor] = dict(runoff_outputs)
        for key in selected:
            if key not in runoff_outputs:
                continue
            value = runoff_outputs[key]
            if not torch.is_tensor(value):
                continue
            transferred[key] = self._transfer_tensor(value, graph)
        return transferred
