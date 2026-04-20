from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_dynamic_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        batch_size, time_steps, channels, nodes = tensor.shape
        return tensor.permute(0, 3, 1, 2).reshape(batch_size, nodes, time_steps, channels)
    if tensor.ndim == 5:
        batch_size, time_steps, channels, y_size, x_size = tensor.shape
        return tensor.permute(0, 3, 4, 1, 2).reshape(batch_size, y_size * x_size, time_steps, channels)
    raise ValueError(
        f"Runoff dynamic input must be [B,T,C,N] or [B,T,C,Y,X], got {tuple(tensor.shape)}"
    )


def _flatten_static_tensor(tensor: torch.Tensor, *, time_steps: int) -> torch.Tensor:
    if tensor.ndim == 3:
        batch_size, channels, nodes = tensor.shape
        static = tensor.permute(0, 2, 1).reshape(batch_size, nodes, channels)
        return static.unsqueeze(2).expand(-1, -1, time_steps, -1)
    if tensor.ndim == 4:
        batch_size, channels, y_size, x_size = tensor.shape
        static = tensor.permute(0, 2, 3, 1).reshape(batch_size, y_size * x_size, channels)
        return static.unsqueeze(2).expand(-1, -1, time_steps, -1)
    raise ValueError(
        f"Runoff static input must be [B,C,N] or [B,C,Y,X], got {tuple(tensor.shape)}"
    )


class SpatialLSTMRunoffModel(nn.Module):
    """Predict grid/node runoff from gridded forcing before routing.

    The model treats every grid node as a sequence sample that shares LSTM
    weights. It returns runoff tensors shaped [B, T, N], which the existing
    routing models can consume exactly like compact Routing runoff outputs.
    """

    def __init__(
        self,
        *,
        dynamic_input_keys: Sequence[str] = ("x_forcing_ml",),
        static_input_keys: Sequence[str] = (),
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_keys: Sequence[str] = ("runoff_total",),
        output_activation: str = "softplus",
        input_norm: str | None = "layer_norm",
        node_batch_size: int | None = None,
        sanitize_nonfinite: bool = False,
        feature_clip: float | None = None,
        use_cudnn: bool = True,
    ):
        super().__init__()
        self.dynamic_input_keys = tuple(str(key) for key in dynamic_input_keys)
        self.static_input_keys = tuple(str(key) for key in static_input_keys)
        self.input_keys = tuple(dict.fromkeys((*self.dynamic_input_keys, *self.static_input_keys)))
        self.output_keys = tuple(str(key) for key in output_keys)
        self.output_activation = str(output_activation).lower()
        self.input_norm = None if input_norm in {None, "", "none"} else str(input_norm).lower()
        self.node_batch_size = None if node_batch_size in {None, 0} else int(node_batch_size)
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self.feature_clip = None if feature_clip in {None, 0} else float(feature_clip)
        self.use_cudnn = bool(use_cudnn)

        if not self.dynamic_input_keys:
            raise ValueError("SpatialLSTMRunoffModel requires at least one dynamic input key")
        if not self.output_keys:
            raise ValueError("SpatialLSTMRunoffModel requires at least one output key")

        if self.input_norm == "layer_norm":
            self.norm = nn.LayerNorm(int(input_dim))
        elif self.input_norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError("runoff_model.input_norm must be one of: layer_norm, none")

        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.dropout = float(dropout)
        self.head = nn.Linear(int(hidden_dim), len(self.output_keys))

    def _apply_output_activation(self, values: torch.Tensor) -> torch.Tensor:
        if self.output_activation in {"none", "identity"}:
            return values
        if self.output_activation == "softplus":
            return F.softplus(values)
        if self.output_activation == "relu":
            return F.relu(values)
        raise ValueError("runoff_model.output_activation must be one of: softplus, relu, none")

    def _build_features(self, batch: Mapping[str, Any]) -> torch.Tensor:
        feature_parts: list[torch.Tensor] = []
        time_steps = None
        node_count = None

        for key in self.dynamic_input_keys:
            if key not in batch:
                raise KeyError(f"Runoff model batch is missing dynamic input '{key}'")
            tensor = _flatten_dynamic_tensor(batch[key])
            time_steps = int(tensor.shape[2])
            node_count = int(tensor.shape[1])
            feature_parts.append(tensor)

        if time_steps is None or node_count is None:
            raise ValueError("Runoff model requires at least one temporal feature")

        for key in self.static_input_keys:
            if key not in batch:
                raise KeyError(f"Runoff model batch is missing static input '{key}'")
            static_tensor = _flatten_static_tensor(batch[key], time_steps=time_steps)
            if int(static_tensor.shape[1]) != node_count:
                raise ValueError(
                    f"Runoff static input '{key}' has {int(static_tensor.shape[1])} nodes, "
                    f"but dynamic inputs have {node_count} nodes"
                )
            feature_parts.append(static_tensor)

        features = torch.cat(feature_parts, dim=-1)
        if self.sanitize_nonfinite:
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_clip is not None:
            features = torch.clamp(features, min=-self.feature_clip, max=self.feature_clip)
        return features

    def _predict_feature_chunk(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, time_steps, feature_dim = features.shape
        sequences = features.reshape(batch_size * node_count, time_steps, feature_dim)
        sequences = self.norm(sequences)

        with torch.backends.cudnn.flags(enabled=self.use_cudnn):
            hidden, _ = self.lstm(sequences)
        if self.dropout > 0.0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        predictions = self._apply_output_activation(self.head(hidden))
        predictions = predictions.reshape(batch_size, node_count, time_steps, len(self.output_keys))
        return predictions.permute(0, 2, 1, 3).contiguous()

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        features = self._build_features(batch)
        node_count = int(features.shape[1])
        if self.node_batch_size is None or self.node_batch_size >= node_count:
            predictions = self._predict_feature_chunk(features)
        else:
            chunks = []
            for start in range(0, node_count, self.node_batch_size):
                chunk = features[:, start : start + self.node_batch_size].contiguous()
                chunks.append(self._predict_feature_chunk(chunk))
            predictions = torch.cat(chunks, dim=2)

        return {
            key: predictions[..., output_index]
            for output_index, key in enumerate(self.output_keys)
        }


class SpatialTemporalConvRunoffModel(nn.Module):
    """Predict runoff with causal temporal convolutions shared across nodes."""

    def __init__(
        self,
        *,
        dynamic_input_keys: Sequence[str] = ("x_forcing_ml",),
        static_input_keys: Sequence[str] = (),
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
        output_keys: Sequence[str] = ("runoff_total",),
        output_activation: str = "softplus",
        sanitize_nonfinite: bool = False,
        feature_clip: float | None = None,
        node_batch_size: int | None = None,
    ):
        super().__init__()
        self.dynamic_input_keys = tuple(str(key) for key in dynamic_input_keys)
        self.static_input_keys = tuple(str(key) for key in static_input_keys)
        self.input_keys = tuple(dict.fromkeys((*self.dynamic_input_keys, *self.static_input_keys)))
        self.output_keys = tuple(str(key) for key in output_keys)
        self.output_activation = str(output_activation).lower()
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self.feature_clip = None if feature_clip in {None, 0} else float(feature_clip)
        self.node_batch_size = None if node_batch_size in {None, 0} else int(node_batch_size)
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError("runoff_model.kernel_size must be positive")

        layers = []
        in_channels = int(input_dim)
        for layer_idx in range(int(num_layers)):
            dilation = 2**layer_idx
            conv = nn.Conv1d(
                in_channels,
                int(hidden_dim),
                kernel_size=self.kernel_size,
                dilation=dilation,
            )
            layers.append(
                nn.ModuleDict(
                    {
                        "conv": conv,
                        "norm": nn.GroupNorm(num_groups=1, num_channels=int(hidden_dim)),
                    }
                )
            )
            in_channels = int(hidden_dim)
        self.layers = nn.ModuleList(layers)
        self.dropout = float(dropout)
        self.head = nn.Conv1d(in_channels, len(self.output_keys), kernel_size=1)

        if not self.dynamic_input_keys:
            raise ValueError("SpatialTemporalConvRunoffModel requires at least one dynamic input key")
        if not self.output_keys:
            raise ValueError("SpatialTemporalConvRunoffModel requires at least one output key")

    def _apply_output_activation(self, values: torch.Tensor) -> torch.Tensor:
        if self.output_activation in {"none", "identity"}:
            return values
        if self.output_activation == "softplus":
            return F.softplus(values)
        if self.output_activation == "relu":
            return F.relu(values)
        raise ValueError("runoff_model.output_activation must be one of: softplus, relu, none")

    def _build_features(self, batch: Mapping[str, Any]) -> torch.Tensor:
        feature_parts: list[torch.Tensor] = []
        time_steps = None
        node_count = None

        for key in self.dynamic_input_keys:
            if key not in batch:
                raise KeyError(f"Runoff model batch is missing dynamic input '{key}'")
            tensor = _flatten_dynamic_tensor(batch[key])
            time_steps = int(tensor.shape[2])
            node_count = int(tensor.shape[1])
            feature_parts.append(tensor)

        if time_steps is None or node_count is None:
            raise ValueError("Runoff model requires at least one temporal feature")

        for key in self.static_input_keys:
            if key not in batch:
                raise KeyError(f"Runoff model batch is missing static input '{key}'")
            static_tensor = _flatten_static_tensor(batch[key], time_steps=time_steps)
            if int(static_tensor.shape[1]) != node_count:
                raise ValueError(
                    f"Runoff static input '{key}' has {int(static_tensor.shape[1])} nodes, "
                    f"but dynamic inputs have {node_count} nodes"
                )
            feature_parts.append(static_tensor)

        features = torch.cat(feature_parts, dim=-1)
        if self.sanitize_nonfinite:
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.feature_clip is not None:
            features = torch.clamp(features, min=-self.feature_clip, max=self.feature_clip)
        return features

    def _predict_feature_chunk(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, time_steps, feature_dim = features.shape
        x = features.reshape(batch_size * node_count, time_steps, feature_dim).transpose(1, 2)
        for layer in self.layers:
            conv = layer["conv"]
            pad_left = int(conv.dilation[0]) * (int(conv.kernel_size[0]) - 1)
            residual = x
            x = F.pad(x, (pad_left, 0))
            x = conv(x)
            x = layer["norm"](x)
            x = F.silu(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:
                x = x + residual
        predictions = self._apply_output_activation(self.head(x))
        predictions = predictions.transpose(1, 2).reshape(
            batch_size,
            node_count,
            time_steps,
            len(self.output_keys),
        )
        return predictions.permute(0, 2, 1, 3).contiguous()

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        features = self._build_features(batch)
        node_count = int(features.shape[1])
        if self.node_batch_size is None or self.node_batch_size >= node_count:
            predictions = self._predict_feature_chunk(features)
        else:
            chunks = []
            for start in range(0, node_count, self.node_batch_size):
                chunk = features[:, start : start + self.node_batch_size].contiguous()
                chunks.append(self._predict_feature_chunk(chunk))
            predictions = torch.cat(chunks, dim=2)

        return {
            key: predictions[..., output_index]
            for output_index, key in enumerate(self.output_keys)
        }
