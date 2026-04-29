from __future__ import annotations

import json
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


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def _coerce_name_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return tuple(str(item) for item in parsed)
            except Exception:
                pass
        return (text,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


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
        weight_strategy: str = "stored",
        source_feature_names_key: str = "runoff_source_feature_names",
        target_feature_key: str = "node_features",
        target_feature_names_key: str = "node_feature_names",
        cell_area_feature_name: str = "cell_area_m2",
        distance_feature_name: str = "distance_to_flowpath_m",
        source_elevation_feature_name: str = "elevation",
        target_elevation_feature_names: Sequence[str] | None = None,
        preserve_base_weight_sum: bool = True,
        distance_scale_m: float | None = None,
        sanitize_nonfinite: bool = True,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"fixed", "neural", "learned"}:
            raise ValueError("runoff_transfer.mode must be one of: fixed, neural")
        self.weight_strategy = str(weight_strategy or "stored").lower()
        if self.weight_strategy not in {
            "stored",
            "cell_area",
            "inverse_distance",
            "exp_distance",
            "downhill",
            "downhill_distance",
        }:
            raise ValueError(
                "runoff_transfer.weight_strategy must be one of: "
                "stored, cell_area, inverse_distance, exp_distance, downhill, downhill_distance"
            )
        self.graph_key = str(graph_key)
        self.output_keys = tuple(str(key) for key in output_keys or ())
        self.source_index_key = str(source_index_key)
        self.source_flat_index_key = str(source_flat_index_key)
        self.target_index_key = str(target_index_key)
        self.weight_key = str(weight_key)
        self.source_feature_key = str(source_feature_key)
        self.source_feature_names_key = str(source_feature_names_key)
        self.target_feature_key = str(target_feature_key)
        self.target_feature_names_key = str(target_feature_names_key)
        self.cell_area_feature_name = str(cell_area_feature_name)
        self.distance_feature_name = str(distance_feature_name)
        self.source_elevation_feature_name = str(source_elevation_feature_name)
        target_elevation_feature_names = target_elevation_feature_names or ("node_dem_elevation_m", "mean.elevation")
        self.target_elevation_feature_names = tuple(str(value) for value in target_elevation_feature_names)
        self.preserve_base_weight_sum = bool(preserve_base_weight_sum)
        self.distance_scale_m = None if distance_scale_m in {None, 0} else float(distance_scale_m)
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

    def _feature_index(self, names: Sequence[str], requested_names: Sequence[str]) -> int | None:
        normalized = {_normalize_name(name): idx for idx, name in enumerate(names)}
        for requested in requested_names:
            if _normalize_name(requested) in normalized:
                return int(normalized[_normalize_name(requested)])
        return None

    def _metadata(self, graph: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = graph.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def _source_feature_vector(
        self,
        graph: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        requested_names: Sequence[str],
    ) -> torch.Tensor | None:
        raw = _resolve_graph_tensor(graph, (self.source_feature_key, "runoff_source_features"))
        if raw is None:
            return None
        features = torch.as_tensor(raw, dtype=dtype, device=device)
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        names_raw = graph.get(self.source_feature_names_key, graph.get("runoff_source_feature_names", ()))
        names = _coerce_name_sequence(names_raw)
        index = self._feature_index(names, requested_names)
        if index is None:
            return None
        return features[:, index].reshape(-1)

    def _target_feature_vector(
        self,
        graph: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        requested_names: Sequence[str],
    ) -> torch.Tensor | None:
        raw = _resolve_graph_tensor(graph, (self.target_feature_key, "node_features"))
        if raw is None:
            return None
        features = torch.as_tensor(raw, dtype=dtype, device=device)
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        names_raw = graph.get(self.target_feature_names_key)
        if names_raw is None:
            names_raw = self._metadata(graph).get(
                self.target_feature_names_key,
                self._metadata(graph).get("node_feature_names", ()),
            )
        names = _coerce_name_sequence(names_raw)
        index = self._feature_index(names, requested_names)
        if index is None:
            return None
        return features[:, index].reshape(-1)

    def _cell_scale_m(
        self,
        graph: Mapping[str, Any],
        base_weights: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cell_area = self._source_feature_vector(
            graph,
            device=device,
            dtype=dtype,
            requested_names=(self.cell_area_feature_name,),
        )
        if cell_area is None:
            cell_area = base_weights
        return torch.sqrt(torch.clamp(cell_area, min=torch.finfo(dtype).eps))

    def _preserve_weight_sum(
        self,
        base_weights: torch.Tensor,
        candidate_weights: torch.Tensor,
        target_index: torch.Tensor,
        *,
        num_targets: int,
    ) -> torch.Tensor:
        eps = torch.finfo(candidate_weights.dtype).eps
        base_sum = torch.zeros(num_targets, dtype=base_weights.dtype, device=base_weights.device)
        cand_sum = torch.zeros(num_targets, dtype=candidate_weights.dtype, device=candidate_weights.device)
        base_sum.scatter_add_(0, target_index, base_weights)
        cand_sum.scatter_add_(0, target_index, candidate_weights)

        fallback_targets = cand_sum <= eps
        if bool(fallback_targets.any()):
            candidate_weights = torch.where(fallback_targets.index_select(0, target_index), base_weights, candidate_weights)
            cand_sum.zero_()
            cand_sum.scatter_add_(0, target_index, candidate_weights)

        scale = base_sum / cand_sum.clamp_min(eps)
        return candidate_weights * scale.index_select(0, target_index)

    def _resolve_transfer_weights(
        self,
        graph: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        base_weights: torch.Tensor,
        target_index: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        strategy = self.weight_strategy
        if strategy == "stored":
            return base_weights

        cell_area = self._source_feature_vector(
            graph,
            device=device,
            dtype=dtype,
            requested_names=(self.cell_area_feature_name,),
        )
        if cell_area is None:
            cell_area = base_weights

        candidate_weights: torch.Tensor
        if strategy == "cell_area":
            candidate_weights = cell_area
        elif strategy in {"inverse_distance", "exp_distance"}:
            distance = self._source_feature_vector(
                graph,
                device=device,
                dtype=dtype,
                requested_names=(self.distance_feature_name,),
            )
            if distance is None:
                raise KeyError(
                    f"runoff_transfer.weight_strategy='{strategy}' requires source feature "
                    f"'{self.distance_feature_name}' in routing_graph['{self.source_feature_key}']"
                )
            distance = torch.maximum(
                distance,
                self._cell_scale_m(graph, base_weights, device=device, dtype=dtype),
            )
            if strategy == "inverse_distance":
                factor = 1.0 / distance
            else:
                scale = self.distance_scale_m
                if scale is None or scale <= 0.0:
                    finite_distance = distance[torch.isfinite(distance)]
                    scale = float(finite_distance.median().item()) if int(finite_distance.numel()) else 1.0
                factor = torch.exp(-distance / max(scale, torch.finfo(dtype).eps))
            candidate_weights = cell_area * factor
        elif strategy in {"downhill", "downhill_distance"}:
            source_elevation = self._source_feature_vector(
                graph,
                device=device,
                dtype=dtype,
                requested_names=(self.source_elevation_feature_name,),
            )
            if source_elevation is None:
                raise KeyError(
                    f"runoff_transfer.weight_strategy='{strategy}' requires source feature "
                    f"'{self.source_elevation_feature_name}' in routing_graph['{self.source_feature_key}']"
                )
            target_elevation = self._target_feature_vector(
                graph,
                device=device,
                dtype=dtype,
                requested_names=self.target_elevation_feature_names,
            )
            if target_elevation is None:
                raise KeyError(
                    f"runoff_transfer.weight_strategy='{strategy}' requires a node feature named one of "
                    f"{list(self.target_elevation_feature_names)} in routing_graph['{self.target_feature_key}']"
                )
            elevation_drop = torch.clamp(source_elevation - target_elevation.index_select(0, target_index), min=0.0)
            if strategy == "downhill":
                factor = elevation_drop
            else:
                distance = self._source_feature_vector(
                    graph,
                    device=device,
                    dtype=dtype,
                    requested_names=(self.distance_feature_name,),
                )
                if distance is None:
                    raise KeyError(
                        f"runoff_transfer.weight_strategy='{strategy}' requires source feature "
                        f"'{self.distance_feature_name}' in routing_graph['{self.source_feature_key}']"
                    )
                distance = torch.maximum(
                    distance,
                    self._cell_scale_m(graph, base_weights, device=device, dtype=dtype),
                )
                factor = elevation_drop / distance
            candidate_weights = cell_area * factor
        else:
            raise ValueError(f"Unsupported runoff transfer weight_strategy '{strategy}'")

        if self.sanitize_nonfinite:
            candidate_weights = torch.nan_to_num(candidate_weights, nan=0.0, posinf=0.0, neginf=0.0)
        if self.preserve_base_weight_sum:
            candidate_weights = self._preserve_weight_sum(
                base_weights,
                candidate_weights,
                target_index,
                num_targets=num_targets,
            )
        return candidate_weights

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
        weights = self._resolve_transfer_weights(
            graph,
            device=tensor.device,
            dtype=tensor.dtype,
            base_weights=base_weights,
            target_index=target_index,
            num_targets=num_targets,
        )
        weights = weights * self._source_multipliers(
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
