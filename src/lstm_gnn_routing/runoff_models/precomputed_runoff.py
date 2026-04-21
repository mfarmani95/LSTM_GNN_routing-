from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


def _flatten_channel_names(batch: Mapping[str, Any], input_keys: Sequence[str]) -> list[str]:
    info = None
    x_info = batch.get("x_info")
    if isinstance(x_info, list) and x_info:
        info = x_info[0]
    elif isinstance(x_info, Mapping):
        info = x_info

    names: list[str] = []
    if isinstance(info, Mapping):
        for key in input_keys:
            raw = info.get(f"{key}_names", [])
            names.extend(str(value) for value in raw)
    return names


class PrecomputedRunoffModel(nn.Module):
    """Return precomputed runoff channels as named runoff outputs.

    This lets the routing stack consume externally generated runoff fields
    without running an internal runoff generator such as an LSTM.
    """

    def __init__(
        self,
        *,
        dynamic_input_keys: Sequence[str] = ("x_routing_dynamic",),
        output_keys: Sequence[str] = ("RUNSF", "RUNSB"),
        input_channel_names: Sequence[str] | None = None,
        sanitize_nonfinite: bool = True,
    ):
        super().__init__()
        self.dynamic_input_keys = tuple(str(key) for key in dynamic_input_keys)
        self.static_input_keys: tuple[str, ...] = ()
        self.input_keys = self.dynamic_input_keys
        self.output_keys = tuple(str(key) for key in output_keys)
        self.input_channel_names = tuple(str(value) for value in (input_channel_names or ()))
        self.sanitize_nonfinite = bool(sanitize_nonfinite)

        if not self.dynamic_input_keys:
            raise ValueError("PrecomputedRunoffModel requires at least one dynamic input key")
        if not self.output_keys:
            raise ValueError("PrecomputedRunoffModel requires at least one output key")

    @staticmethod
    def _select_channel(tensor: torch.Tensor, index: int) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor[:, :, index]
        if tensor.ndim == 5:
            return tensor[:, :, index]
        raise ValueError(
            f"Precomputed runoff input must be [B,T,C,N] or [B,T,C,Y,X], got {tuple(tensor.shape)}"
        )

    def _resolve_channel_index(self, output_key: str, channel_names: Sequence[str]) -> int:
        normalized = {str(name).lower(): idx for idx, name in enumerate(channel_names)}
        key_lower = str(output_key).lower()
        if key_lower in normalized:
            return int(normalized[key_lower])
        raise KeyError(
            f"Precomputed runoff output '{output_key}' is missing from available channels {list(channel_names)}"
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        tensors = []
        for key in self.dynamic_input_keys:
            if key not in batch:
                raise KeyError(f"Precomputed runoff batch is missing dynamic input '{key}'")
            tensors.append(batch[key])

        if len(tensors) != 1:
            raise ValueError(
                "PrecomputedRunoffModel currently expects exactly one dynamic runoff tensor "
                f"but received {len(tensors)} keys: {self.dynamic_input_keys}"
            )
        tensor = tensors[0]
        if tensor.ndim not in {4, 5}:
            raise ValueError(
                f"Precomputed runoff input must be [B,T,C,N] or [B,T,C,Y,X], got {tuple(tensor.shape)}"
            )
        if self.sanitize_nonfinite:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        channel_names = list(self.input_channel_names)
        if not channel_names:
            channel_names = _flatten_channel_names(batch, self.dynamic_input_keys)
        if not channel_names:
            channel_names = [f"channel_{idx}" for idx in range(int(tensor.shape[2]))]

        outputs: dict[str, torch.Tensor] = {}
        for key in self.output_keys:
            if str(key).lower() == "runoff_total":
                if "runsf" in {name.lower() for name in channel_names} and "runsb" in {name.lower() for name in channel_names}:
                    runsf = self._select_channel(tensor, self._resolve_channel_index("RUNSF", channel_names))
                    runsb = self._select_channel(tensor, self._resolve_channel_index("RUNSB", channel_names))
                    outputs[key] = runsf + runsb
                    continue
            outputs[key] = self._select_channel(tensor, self._resolve_channel_index(key, channel_names))
        return outputs
