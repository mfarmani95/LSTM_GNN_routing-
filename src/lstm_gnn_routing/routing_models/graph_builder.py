from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.auto import tqdm


_D8_OFFSETS = {
    "arcgis": {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    },
    "esri": {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    },
    "1to8": {
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (1, -1),
        5: (0, -1),
        6: (-1, -1),
        7: (-1, 0),
        8: (-1, 1),
    },
}


def _flatten_index(y: int, x: int, x_size: int) -> int:
    return y * x_size + x


def _as_2d_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2-D array, got shape {tuple(array.shape)}")
    return array


def _resolve_active_mask(mask_array: np.ndarray | None, mask_spec: Mapping[str, Any] | None) -> np.ndarray:
    if mask_array is None:
        if mask_spec:
            raise ValueError("routing.graph.mask was provided but no mask array could be resolved")
        return np.ones((0, 0), dtype=bool)

    mask = np.isfinite(mask_array)
    if not mask_spec:
        return mask & (mask_array != 0)

    include_values = mask_spec.get("include_values")
    exclude_values = mask_spec.get("exclude_values")
    min_value = mask_spec.get("min_value")
    max_value = mask_spec.get("max_value")
    non_nan_only = bool(mask_spec.get("non_nan_only", False) or mask_spec.get("finite_only", False))

    if include_values is not None:
        mask &= np.isin(mask_array, np.asarray(include_values))
    elif exclude_values is not None:
        mask &= ~np.isin(mask_array, np.asarray(exclude_values))
    elif not non_nan_only:
        mask &= mask_array != 0
        if min_value is not None:
            mask &= mask_array >= float(min_value)
        if max_value is not None:
            mask &= mask_array <= float(max_value)
    else:
        if min_value is not None:
            mask &= mask_array >= float(min_value)
        if max_value is not None:
            mask &= mask_array <= float(max_value)

    if bool(mask_spec.get("invert", False)):
        mask = ~mask
    return mask


def _unique_edge_index(edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    unique = sorted(set(edges))
    return torch.tensor(unique, dtype=torch.long).t().contiguous()


def _unique_edges_with_attributes(
    edges: Sequence[tuple[int, int]],
    edge_attributes: Sequence[Sequence[float]] | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long), None
    if edge_attributes is None:
        return _unique_edge_index(edges), None
    if len(edges) != len(edge_attributes):
        raise ValueError("edges and edge_attributes must have the same length")

    merged: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}
    for edge, attrs in zip(edges, edge_attributes):
        values = np.asarray(attrs, dtype=np.float64)
        if edge in merged:
            merged[edge] += values
            counts[edge] += 1
        else:
            merged[edge] = values.copy()
            counts[edge] = 1

    unique = sorted(merged)
    edge_index = torch.tensor(unique, dtype=torch.long).t().contiguous()
    attr = np.stack([merged[edge] / float(counts[edge]) for edge in unique], axis=0).astype(np.float32)
    attr = np.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_index, torch.as_tensor(attr, dtype=torch.float32)


def _coerce_dbf_value(raw: str, field_type: str) -> Any:
    text = raw.strip()
    if text == "":
        return None
    if field_type in {"N", "F", "B"}:
        try:
            value = float(text)
        except ValueError:
            return text
        return int(value) if value.is_integer() else value
    if field_type == "L":
        return text.upper() in {"Y", "T", "1"}
    return text


def _read_dbf_records(dbf_path: Path) -> list[dict[str, Any]]:
    data = dbf_path.read_bytes()
    if len(data) < 32:
        raise ValueError(f"Invalid DBF file: {dbf_path}")
    record_count = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]

    fields: list[tuple[str, str, int]] = []
    offset = 32
    while offset < len(data) and data[offset] != 0x0D:
        desc = data[offset : offset + 32]
        name = desc[:11].split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
        field_type = chr(desc[11])
        field_len = int(desc[16])
        if name:
            fields.append((name, field_type, field_len))
        offset += 32

    records: list[dict[str, Any]] = []
    for record_idx in range(record_count):
        start = header_len + record_idx * record_len
        record = data[start : start + record_len]
        if not record or record[0:1] == b"*":
            records.append({})
            continue
        pos = 1
        values: dict[str, Any] = {}
        for name, field_type, field_len in fields:
            raw = record[pos : pos + field_len].decode("latin1", errors="ignore")
            values[name] = _coerce_dbf_value(raw, field_type)
            pos += field_len
        records.append(values)
    return records


def _read_polyline_shapefile(file_path: Path) -> list[dict[str, Any]]:
    shp_path = Path(file_path)
    if shp_path.suffix.lower() != ".shp":
        shp_path = shp_path.with_suffix(".shp")
    dbf_path = shp_path.with_suffix(".dbf")
    dbf_records = _read_dbf_records(dbf_path) if dbf_path.exists() else []
    data = shp_path.read_bytes()
    if len(data) < 100:
        raise ValueError(f"Invalid shapefile: {shp_path}")

    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type not in {3, 13, 23}:
        raise ValueError(
            f"Flowline shapefile must contain PolyLine/PolyLineZ/PolyLineM geometries, got shape type {shape_type}"
        )

    records: list[dict[str, Any]] = []
    offset = 100
    while offset + 8 <= len(data):
        record_number, content_words = struct.unpack(">2i", data[offset : offset + 8])
        content_size = int(content_words) * 2
        content = data[offset + 8 : offset + 8 + content_size]
        offset += 8 + content_size
        if len(content) < 4:
            continue

        record_shape_type = struct.unpack("<i", content[:4])[0]
        if record_shape_type == 0:
            continue
        if record_shape_type not in {3, 13, 23}:
            continue
        if len(content) < 44:
            continue

        num_parts, num_points = struct.unpack("<2i", content[36:44])
        parts_start = 44
        points_start = parts_start + 4 * int(num_parts)
        points_end = points_start + 16 * int(num_points)
        if points_end > len(content):
            continue

        parts = list(struct.unpack("<" + "i" * int(num_parts), content[parts_start:points_start]))
        points = np.frombuffer(content[points_start:points_end], dtype="<f8").reshape(int(num_points), 2)
        attrs = dbf_records[record_number - 1] if 0 <= record_number - 1 < len(dbf_records) else {}

        geometry_parts: list[np.ndarray] = []
        for part_idx, start_idx in enumerate(parts):
            end_idx = parts[part_idx + 1] if part_idx + 1 < len(parts) else int(num_points)
            if end_idx - start_idx >= 2:
                geometry_parts.append(points[start_idx:end_idx].astype(np.float64, copy=True))
        if geometry_parts:
            records.append({"parts": geometry_parts, "attributes": attrs, "record_number": record_number})
    return records


def _numeric_attribute(record: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = record.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _compact_graph_indices(
    *,
    edge_index: torch.Tensor,
    gauge_index: torch.Tensor | None,
    active_mask: np.ndarray,
    x_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_flat = np.flatnonzero(active_mask.reshape(-1))
    flat_index = torch.as_tensor(active_flat, dtype=torch.long)
    num_active = int(flat_index.numel())

    if num_active == 0:
        raise ValueError("Routing graph mask produced zero active nodes")

    full_node_count = int(active_mask.size)
    reindex = torch.full((full_node_count,), -1, dtype=torch.long)
    reindex[flat_index] = torch.arange(num_active, dtype=torch.long)

    compact_edge_index = edge_index
    if compact_edge_index.numel():
        compact_edge_index = reindex[compact_edge_index]
        if torch.any(compact_edge_index < 0):
            raise ValueError("Routing graph contains edges that point to inactive nodes after masking")

    compact_gauge_index = gauge_index
    if compact_gauge_index is not None:
        compact_gauge_index = reindex[compact_gauge_index]
        if compact_gauge_index.numel() and torch.any(compact_gauge_index < 0):
            raise ValueError("Gauge indices resolve to inactive routing nodes after masking")

    node_y = flat_index // int(x_size)
    node_x = flat_index % int(x_size)
    return compact_edge_index, compact_gauge_index, flat_index, node_y, node_x


def _compact_node_features(
    node_feature_array: np.ndarray | None,
    *,
    active_mask: np.ndarray,
) -> torch.Tensor | None:
    if node_feature_array is None:
        return None

    array = np.asarray(node_feature_array)
    if array.ndim == 2:
        flat = array.reshape(-1, 1)
    elif array.ndim == 3:
        y_size, x_size, channels = array.shape
        flat = array.reshape(y_size * x_size, channels)
    else:
        raise ValueError(
            f"routing.graph.node_features must resolve to [Y,X] or [Y,X,C], got {tuple(array.shape)}"
        )

    active_flat = np.flatnonzero(active_mask.reshape(-1))
    compact = flat[active_flat]
    compact = np.nan_to_num(compact, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(compact, dtype=torch.float32)


def _compute_compact_edge_features(
    *,
    edge_index: torch.Tensor,
    flat_index: torch.Tensor,
    x_size: int,
    elevation_array: np.ndarray | None,
    y2d: np.ndarray | None,
    x2d: np.ndarray | None,
    feature_names: Sequence[str] | None,
    weight_feature: str | None,
    weight_normalization: str | None,
) -> tuple[torch.Tensor | None, list[str], torch.Tensor | None]:
    requested_features = [str(name) for name in (feature_names or [])]
    normalized_weight_feature = None if weight_feature in {None, ""} else str(weight_feature)
    all_required = list(dict.fromkeys(requested_features + ([normalized_weight_feature] if normalized_weight_feature else [])))

    if edge_index.numel() == 0 or not all_required:
        return None, requested_features, None

    source_flat = flat_index.index_select(0, edge_index[0]).cpu().numpy()
    target_flat = flat_index.index_select(0, edge_index[1]).cpu().numpy()
    source_y = source_flat // int(x_size)
    source_x = source_flat % int(x_size)
    target_y = target_flat // int(x_size)
    target_x = target_flat % int(x_size)

    if x2d is not None and y2d is not None:
        dx = np.asarray(x2d[target_y, target_x] - x2d[source_y, source_x], dtype=np.float32)
        dy = np.asarray(y2d[target_y, target_x] - y2d[source_y, source_x], dtype=np.float32)
    else:
        dx = np.asarray(target_x - source_x, dtype=np.float32)
        dy = np.asarray(target_y - source_y, dtype=np.float32)
    distance = np.sqrt(dx**2 + dy**2).astype(np.float32)

    if elevation_array is not None:
        elevation = np.asarray(elevation_array, dtype=np.float32)
        source_elevation = np.asarray(elevation[source_y, source_x], dtype=np.float32)
        target_elevation = np.asarray(elevation[target_y, target_x], dtype=np.float32)
        elevation_drop = source_elevation - target_elevation
    else:
        source_elevation = target_elevation = elevation_drop = None

    values_by_name: dict[str, np.ndarray] = {}
    for name in all_required:
        if name == "distance":
            values_by_name[name] = distance
        elif name == "elevation_drop":
            if elevation_drop is None:
                raise ValueError("edge feature 'elevation_drop' requires an elevation array")
            values_by_name[name] = elevation_drop.astype(np.float32)
        elif name == "slope":
            if elevation_drop is None:
                raise ValueError("edge feature 'slope' requires an elevation array")
            values_by_name[name] = (elevation_drop / np.maximum(distance, 1.0e-6)).astype(np.float32)
        elif name in {"travel_time_kirpich", "inverse_travel_time_kirpich"}:
            if elevation_drop is None:
                raise ValueError(f"edge feature '{name}' requires an elevation array")
            if x2d is None or y2d is None:
                raise ValueError(f"edge feature '{name}' requires projected x2d/y2d coordinates for distance in meters")
            positive_slope = np.maximum(elevation_drop / np.maximum(distance, 1.0e-6), 1.0e-6)
            distance_m = np.maximum(distance, 1.0)
            travel_time_minutes = 0.01947 * np.power(distance_m, 0.77) * np.power(positive_slope, -0.385)
            travel_time_minutes = travel_time_minutes.astype(np.float32)
            if name == "travel_time_kirpich":
                values_by_name[name] = travel_time_minutes
            else:
                values_by_name[name] = (1.0 / np.maximum(travel_time_minutes, 1.0e-6)).astype(np.float32)
        elif name == "source_elevation":
            if source_elevation is None:
                raise ValueError("edge feature 'source_elevation' requires an elevation array")
            values_by_name[name] = source_elevation.astype(np.float32)
        elif name == "target_elevation":
            if target_elevation is None:
                raise ValueError("edge feature 'target_elevation' requires an elevation array")
            values_by_name[name] = target_elevation.astype(np.float32)
        elif name == "dx":
            values_by_name[name] = dx.astype(np.float32)
        elif name == "dy":
            values_by_name[name] = dy.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported derived edge feature '{name}'. "
                "Choose from: distance, elevation_drop, slope, travel_time_kirpich, "
                "inverse_travel_time_kirpich, source_elevation, target_elevation, dx, dy."
            )

    edge_attr = None
    if requested_features:
        stacked = np.stack([values_by_name[name] for name in requested_features], axis=1).astype(np.float32)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.as_tensor(stacked, dtype=torch.float32)

    edge_weight = None
    if normalized_weight_feature is not None:
        base = np.asarray(values_by_name[normalized_weight_feature], dtype=np.float32)
        if normalized_weight_feature in {"distance", "travel_time_kirpich"}:
            weights = np.where(base > 0.0, 1.0 / np.maximum(base, 1.0e-6), 0.0)
        else:
            weights = np.maximum(base, 0.0)

        normalization_key = str(weight_normalization or "none").lower()
        if normalization_key in {"source_sum", "row_sum", "outgoing_sum"}:
            source_compact = edge_index[0].cpu().numpy()
            denom = np.zeros(int(flat_index.numel()), dtype=np.float32)
            np.add.at(denom, source_compact, weights)
            valid = denom[source_compact] > 0.0
            weights = np.where(valid, weights / np.maximum(denom[source_compact], 1.0e-12), 0.0)
        elif normalization_key not in {"none", "", "null"}:
            raise ValueError(
                f"Unsupported edge weight normalization '{weight_normalization}'. "
                "Choose from: none, source_sum."
            )

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        edge_weight = torch.as_tensor(weights, dtype=torch.float32)

    return edge_attr, requested_features, edge_weight


def export_routing_graph_netcdf(payload: Mapping[str, Any], file_path: str | Path) -> Path:
    edge_index = torch.as_tensor(payload["edge_index"], dtype=torch.long).cpu().numpy()
    flat_index = torch.as_tensor(payload["flat_index"], dtype=torch.long).cpu().numpy()
    node_y = torch.as_tensor(payload["node_y"], dtype=torch.long).cpu().numpy()
    node_x = torch.as_tensor(payload["node_x"], dtype=torch.long).cpu().numpy()

    data_vars: dict[str, Any] = {
        "edge_source": (("edge",), edge_index[0]),
        "edge_target": (("edge",), edge_index[1]),
        "flat_index": (("node",), flat_index),
        "node_y": (("node",), node_y),
        "node_x": (("node",), node_x),
    }
    coords: dict[str, Any] = {
        "edge": np.arange(edge_index.shape[1], dtype=np.int32),
        "node": np.arange(flat_index.shape[0], dtype=np.int32),
    }

    edge_attr = payload.get("edge_attr")
    edge_feature_names = list(payload.get("edge_feature_names", []) or [])
    if edge_attr is not None:
        edge_attr_np = torch.as_tensor(edge_attr, dtype=torch.float32).cpu().numpy()
        coords["edge_feature"] = (
            np.asarray(edge_feature_names, dtype=str)
            if edge_feature_names
            else np.asarray([f"edge_feature_{idx}" for idx in range(edge_attr_np.shape[1])], dtype=str)
        )
        data_vars["edge_attr"] = (("edge", "edge_feature"), edge_attr_np)

    edge_weight = payload.get("edge_weight")
    if edge_weight is not None:
        data_vars["edge_weight"] = (("edge",), torch.as_tensor(edge_weight, dtype=torch.float32).cpu().numpy())

    node_features = payload.get("node_features")
    if node_features is not None:
        node_features_np = torch.as_tensor(node_features, dtype=torch.float32).cpu().numpy()
        coords["node_feature"] = np.asarray(
            [f"node_feature_{idx}" for idx in range(node_features_np.shape[1])],
            dtype=str,
        )
        data_vars["node_features"] = (("node", "node_feature"), node_features_np)

    gauge_index = payload.get("gauge_index")
    if gauge_index is not None:
        gauge_index_np = torch.as_tensor(gauge_index, dtype=torch.long).cpu().numpy()
        coords["gauge"] = np.arange(gauge_index_np.shape[0], dtype=np.int32)
        data_vars["gauge_index"] = (("gauge",), gauge_index_np)
        gauge_ids = list(payload.get("gauge_ids", []) or [])
        if gauge_ids:
            data_vars["gauge_id"] = (("gauge",), np.asarray([str(value) for value in gauge_ids], dtype=str))

    runoff_target_index = payload.get("runoff_target_index")
    if runoff_target_index is not None:
        target_np = torch.as_tensor(runoff_target_index, dtype=torch.long).cpu().numpy()
        coords["runoff_source"] = np.arange(target_np.shape[0], dtype=np.int32)
        data_vars["runoff_target_index"] = (("runoff_source",), target_np)
        runoff_source_index = payload.get("runoff_source_index")
        if runoff_source_index is not None:
            data_vars["runoff_source_index"] = (
                ("runoff_source",),
                torch.as_tensor(runoff_source_index, dtype=torch.long).cpu().numpy(),
            )
        runoff_source_flat_index = payload.get("runoff_source_flat_index")
        if runoff_source_flat_index is not None:
            data_vars["runoff_source_flat_index"] = (
                ("runoff_source",),
                torch.as_tensor(runoff_source_flat_index, dtype=torch.long).cpu().numpy(),
            )
        runoff_source_weight = payload.get("runoff_source_weight")
        if runoff_source_weight is not None:
            data_vars["runoff_source_weight"] = (
                ("runoff_source",),
                torch.as_tensor(runoff_source_weight, dtype=torch.float32).cpu().numpy(),
            )
        runoff_source_features = payload.get("runoff_source_features")
        if runoff_source_features is not None:
            features_np = torch.as_tensor(runoff_source_features, dtype=torch.float32).cpu().numpy()
            feature_names = list(payload.get("runoff_source_feature_names", []) or [])
            coords["runoff_source_feature"] = np.asarray(
                feature_names
                if feature_names
                else [f"runoff_source_feature_{idx}" for idx in range(features_np.shape[1])],
                dtype=str,
            )
            data_vars["runoff_source_features"] = (("runoff_source", "runoff_source_feature"), features_np)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    metadata = dict(payload.get("metadata", {}) or {})
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict)):
            ds.attrs[key] = json.dumps(value)
        else:
            ds.attrs[key] = value

    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


def _build_grid_edges(
    *,
    grid_shape: Sequence[int],
    active_mask: np.ndarray,
    diagonals: bool,
    directed: bool,
    add_reverse_edges: bool,
    show_progress: bool,
) -> torch.Tensor:
    y_size, x_size = int(grid_shape[0]), int(grid_shape[1])
    if active_mask.shape != (y_size, x_size):
        raise ValueError(
            f"active_mask shape {tuple(active_mask.shape)} must match grid_shape {(y_size, x_size)}"
        )

    if diagonals:
        unique_offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
        full_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    else:
        unique_offsets = [(0, 1), (1, 0)]
        full_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    offsets = full_offsets if directed else unique_offsets
    edges: list[tuple[int, int]] = []
    y_iterator = tqdm(
        range(y_size),
        desc="routing graph:grid edges",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for y in y_iterator:
        for x in range(x_size):
            if not active_mask[y, x]:
                continue
            source = _flatten_index(y, x, x_size)
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= y_size or nx < 0 or nx >= x_size:
                    continue
                if not active_mask[ny, nx]:
                    continue
                target = _flatten_index(ny, nx, x_size)
                edges.append((source, target))
                if not directed and add_reverse_edges:
                    edges.append((target, source))
    return _unique_edge_index(edges)


def _build_flow_direction_edges(
    *,
    flow_direction: np.ndarray,
    active_mask: np.ndarray,
    encoding: str,
    add_reverse_edges: bool,
    show_progress: bool,
) -> torch.Tensor:
    y_size, x_size = flow_direction.shape
    if active_mask.shape != (y_size, x_size):
        raise ValueError(
            f"active_mask shape {tuple(active_mask.shape)} must match flow_direction shape {tuple(flow_direction.shape)}"
        )

    encoding_key = str(encoding).lower()
    if encoding_key not in _D8_OFFSETS:
        raise ValueError(
            f"Unsupported flow_direction_d8 encoding '{encoding}'. "
            f"Choose from {sorted(_D8_OFFSETS)}."
        )
    offsets = _D8_OFFSETS[encoding_key]
    direction = np.asarray(np.rint(flow_direction), dtype=np.int64)
    edges: list[tuple[int, int]] = []
    y_iterator = tqdm(
        range(y_size),
        desc="routing graph:flowdir edges",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for y in y_iterator:
        for x in range(x_size):
            if not active_mask[y, x]:
                continue
            code = int(direction[y, x])
            if code <= 0 or code not in offsets:
                continue
            dy, dx = offsets[code]
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= y_size or nx < 0 or nx >= x_size:
                continue
            if not active_mask[ny, nx]:
                continue
            source = _flatten_index(y, x, x_size)
            target = _flatten_index(ny, nx, x_size)
            edges.append((source, target))
            if add_reverse_edges:
                edges.append((target, source))
    return _unique_edge_index(edges)


def _build_dem_downhill_edges(
    *,
    elevation: np.ndarray,
    active_mask: np.ndarray,
    add_reverse_edges: bool,
    show_progress: bool,
) -> torch.Tensor:
    y_size, x_size = elevation.shape
    if active_mask.shape != (y_size, x_size):
        raise ValueError(
            f"active_mask shape {tuple(active_mask.shape)} must match elevation shape {tuple(elevation.shape)}"
        )

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    edges: list[tuple[int, int]] = []

    y_iterator = tqdm(
        range(y_size),
        desc="routing graph:dem edges",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for y in y_iterator:
        for x in range(x_size):
            if not active_mask[y, x]:
                continue

            source_elev = float(elevation[y, x])
            best_target: tuple[int, int] | None = None
            best_elev = source_elev

            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= y_size or nx < 0 or nx >= x_size:
                    continue
                if not active_mask[ny, nx]:
                    continue

                neighbor_elev = float(elevation[ny, nx])
                if neighbor_elev < best_elev:
                    best_elev = neighbor_elev
                    best_target = (ny, nx)

            if best_target is None:
                continue

            source = _flatten_index(y, x, x_size)
            target = _flatten_index(best_target[0], best_target[1], x_size)
            edges.append((source, target))
            if add_reverse_edges:
                edges.append((target, source))

    return _unique_edge_index(edges)


def _is_rectilinear_grid(y_coords: np.ndarray, x_coords: np.ndarray) -> bool:
    return (
        y_coords.ndim == 2
        and x_coords.ndim == 2
        and np.allclose(x_coords, x_coords[0:1, :], equal_nan=True)
        and np.allclose(y_coords, y_coords[:, 0:1], equal_nan=True)
    )


def _nearest_axis_indices(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis.size == 0:
        raise ValueError("Cannot resolve nearest grid indices from an empty coordinate axis")
    if axis.size == 1:
        return np.zeros(values.shape, dtype=np.int64)

    descending = bool(axis[0] > axis[-1])
    work_axis = axis[::-1] if descending else axis
    insert = np.searchsorted(work_axis, values)
    insert = np.clip(insert, 1, work_axis.size - 1)
    left = work_axis[insert - 1]
    right = work_axis[insert]
    choose_right = np.abs(values - right) < np.abs(values - left)
    idx = np.where(choose_right, insert, insert - 1)
    if descending:
        idx = work_axis.size - 1 - idx
    return idx.astype(np.int64)


def _nearest_grid_indices(
    points: np.ndarray,
    *,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_rectilinear_grid(y_coords, x_coords):
        y_idx = _nearest_axis_indices(points[:, 1], y_coords[:, 0])
        x_idx = _nearest_axis_indices(points[:, 0], x_coords[0, :])
        return y_idx, x_idx

    flat_y = y_coords.reshape(-1)
    flat_x = x_coords.reshape(-1)
    grid_shape = y_coords.shape
    y_idx: list[int] = []
    x_idx: list[int] = []
    for x_value, y_value in points:
        distance = (flat_x - float(x_value)) ** 2 + (flat_y - float(y_value)) ** 2
        flat_idx = int(np.nanargmin(distance))
        y_idx.append(flat_idx // int(grid_shape[1]))
        x_idx.append(flat_idx % int(grid_shape[1]))
    return np.asarray(y_idx, dtype=np.int64), np.asarray(x_idx, dtype=np.int64)


def _estimate_coordinate_step(y_coords: np.ndarray, x_coords: np.ndarray) -> float:
    diffs: list[np.ndarray] = []
    if _is_rectilinear_grid(y_coords, x_coords):
        diffs.extend([np.abs(np.diff(y_coords[:, 0])), np.abs(np.diff(x_coords[0, :]))])
    else:
        diffs.extend([np.abs(np.diff(y_coords, axis=0)).reshape(-1), np.abs(np.diff(x_coords, axis=1)).reshape(-1)])
    positive = np.concatenate([values[np.isfinite(values) & (values > 0.0)] for values in diffs if values.size])
    if positive.size == 0:
        return 1.0
    return float(np.nanmedian(positive))


def _densify_line_part(points: np.ndarray, coordinate_step: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    coordinate_step = max(float(coordinate_step), 1.0e-12)
    dense_parts: list[np.ndarray] = []
    for idx in range(points.shape[0] - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        distance = float(np.linalg.norm(p1 - p0))
        n_steps = max(1, int(np.ceil(distance / coordinate_step)))
        fractions = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        segment = p0[None, :] + fractions[:, None] * (p1 - p0)[None, :]
        dense_parts.append(segment[:-1] if idx + 1 < points.shape[0] - 1 else segment)
    return np.concatenate(dense_parts, axis=0)


def _resolve_flowline_coordinate_grid(
    flowline_records: Sequence[Mapping[str, Any]],
    *,
    coordinate_mode: str,
    lat2d: np.ndarray | None,
    lon2d: np.ndarray | None,
    y2d: np.ndarray | None,
    x2d: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = str(coordinate_mode or "auto").lower()
    if mode in {"latlon", "lonlat", "geographic", "degree", "degrees"}:
        if lat2d is None or lon2d is None:
            raise ValueError("Flowline coordinate_mode='latlon' requires lat2d/lon2d grid coordinates")
        return np.asarray(lat2d, dtype=np.float64), np.asarray(lon2d, dtype=np.float64), "latlon"
    if mode in {"projected", "projection", "lambert", "map", "xy"}:
        if y2d is None or x2d is None:
            raise ValueError("Flowline coordinate_mode='projected' requires x2d/y2d grid coordinates")
        return np.asarray(y2d, dtype=np.float64), np.asarray(x2d, dtype=np.float64), "projected"
    if mode != "auto":
        raise ValueError("flowlines.coordinate_mode must be one of: auto, latlon, projected")

    sample_values: list[float] = []
    for record in flowline_records[: min(25, len(flowline_records))]:
        for part in record.get("parts", []):
            if len(part):
                sample_values.extend([float(part[0, 0]), float(part[0, 1])])
                break
    looks_geographic = sample_values and all(-180.0 <= value <= 180.0 for value in sample_values)
    if looks_geographic and lat2d is not None and lon2d is not None:
        return np.asarray(lat2d, dtype=np.float64), np.asarray(lon2d, dtype=np.float64), "latlon"
    if y2d is not None and x2d is not None:
        return np.asarray(y2d, dtype=np.float64), np.asarray(x2d, dtype=np.float64), "projected"
    if lat2d is not None and lon2d is not None:
        return np.asarray(lat2d, dtype=np.float64), np.asarray(lon2d, dtype=np.float64), "latlon"
    raise ValueError("Flowline builder needs either lat2d/lon2d or x2d/y2d grid coordinates")


def _flowline_part_to_flat_cells(
    part: np.ndarray,
    *,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    active_mask: np.ndarray,
    coordinate_step: float,
    x_size: int,
) -> list[int]:
    dense = _densify_line_part(part, coordinate_step)
    x_min = float(np.nanmin(x_coords)) - coordinate_step
    x_max = float(np.nanmax(x_coords)) + coordinate_step
    y_min = float(np.nanmin(y_coords)) - coordinate_step
    y_max = float(np.nanmax(y_coords)) + coordinate_step
    in_bounds = (
        (dense[:, 0] >= x_min)
        & (dense[:, 0] <= x_max)
        & (dense[:, 1] >= y_min)
        & (dense[:, 1] <= y_max)
    )
    dense = dense[in_bounds]
    if dense.size == 0:
        return []
    y_idx, x_idx = _nearest_grid_indices(dense, y_coords=y_coords, x_coords=x_coords)
    flat_cells: list[int] = []
    previous = None
    for y, x in zip(y_idx.tolist(), x_idx.tolist()):
        if y < 0 or y >= active_mask.shape[0] or x < 0 or x >= active_mask.shape[1]:
            continue
        if not active_mask[y, x]:
            continue
        flat_idx = _flatten_index(int(y), int(x), x_size)
        if previous == flat_idx:
            continue
        flat_cells.append(flat_idx)
        previous = flat_idx
    return flat_cells


def _build_flowline_network_edges(
    *,
    flowline_spec: Mapping[str, Any],
    base_active_mask: np.ndarray,
    lat2d: np.ndarray | None,
    lon2d: np.ndarray | None,
    y2d: np.ndarray | None,
    x2d: np.ndarray | None,
    add_reverse_edges: bool,
    show_progress: bool,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor | None, list[str], dict[str, Any]]:
    file_path = flowline_spec.get("file_path") or flowline_spec.get("path") or flowline_spec.get("shapefile")
    if not file_path:
        raise ValueError("flowline_network builder requires routing.graph.flowlines.file_path")

    records = _read_polyline_shapefile(Path(file_path))
    if not records:
        raise ValueError(f"No polyline records were read from flowline file {file_path}")

    coord_y, coord_x, resolved_mode = _resolve_flowline_coordinate_grid(
        records,
        coordinate_mode=str(flowline_spec.get("coordinate_mode", "auto")),
        lat2d=lat2d,
        lon2d=lon2d,
        y2d=y2d,
        x2d=x2d,
    )
    if coord_y.shape != base_active_mask.shape or coord_x.shape != base_active_mask.shape:
        raise ValueError("Flowline coordinate grid shape must match the active mask shape")

    x_size = int(base_active_mask.shape[1])
    coordinate_step = float(flowline_spec.get("densify_step", 0.0) or 0.0)
    if coordinate_step <= 0.0:
        coordinate_step = _estimate_coordinate_step(coord_y, coord_x) * float(flowline_spec.get("densify_factor", 0.5))

    direction_mode = str(flowline_spec.get("direction", "flowdir")).lower()
    flowdir_column = str(flowline_spec.get("flowdir_column", "FLOWDIR"))
    id_column = str(flowline_spec.get("id_column", "COMID"))
    downstream_column = str(flowline_spec.get("downstream_id_column", "DWNCOMID"))
    connect_downstream = bool(flowline_spec.get("connect_downstream_comid", True))
    attribute_fields = [str(name) for name in flowline_spec.get("attribute_edge_features", []) or []]

    edges: list[tuple[int, int]] = []
    edge_attributes: list[list[float]] | None = [] if attribute_fields else None
    river_mask = np.zeros_like(base_active_mask, dtype=bool)
    segment_cells_by_id: dict[str, list[int]] = {}

    iterator = tqdm(
        records,
        desc="routing graph:flowlines",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for record in iterator:
        attrs = dict(record.get("attributes", {}) or {})
        flowdir_value = str(attrs.get(flowdir_column, "") or "").lower()
        reverse_geometry = direction_mode in {"against_digitized", "reverse"} or (
            direction_mode in {"flowdir", "attribute", "auto"} and "against" in flowdir_value
        )
        segment_cells: list[int] = []
        for raw_part in record.get("parts", []):
            part = raw_part[::-1].copy() if reverse_geometry else raw_part
            cells = _flowline_part_to_flat_cells(
                part,
                y_coords=coord_y,
                x_coords=coord_x,
                active_mask=base_active_mask,
                coordinate_step=coordinate_step,
                x_size=x_size,
            )
            if not cells:
                continue
            segment_cells.extend(cells)
            for flat_idx in cells:
                river_mask.reshape(-1)[flat_idx] = True
            for source, target in zip(cells[:-1], cells[1:]):
                if source == target:
                    continue
                edges.append((source, target))
                if edge_attributes is not None:
                    edge_attributes.append([_numeric_attribute(attrs, name) for name in attribute_fields])
                if add_reverse_edges:
                    edges.append((target, source))
                    if edge_attributes is not None:
                        edge_attributes.append([_numeric_attribute(attrs, name) for name in attribute_fields])

        segment_id = attrs.get(id_column)
        if segment_id is not None and segment_cells:
            segment_cells_by_id[str(int(segment_id)) if isinstance(segment_id, float) and segment_id.is_integer() else str(segment_id)] = segment_cells

    if connect_downstream and downstream_column:
        for record in records:
            attrs = dict(record.get("attributes", {}) or {})
            segment_id = attrs.get(id_column)
            downstream_id = attrs.get(downstream_column)
            if segment_id is None or downstream_id in {None, 0, "0", ""}:
                continue
            source_cells = segment_cells_by_id.get(
                str(int(segment_id)) if isinstance(segment_id, float) and segment_id.is_integer() else str(segment_id)
            )
            target_cells = segment_cells_by_id.get(
                str(int(downstream_id)) if isinstance(downstream_id, float) and downstream_id.is_integer() else str(downstream_id)
            )
            if not source_cells or not target_cells:
                continue
            source = source_cells[-1]
            target = target_cells[0]
            if source == target:
                continue
            edges.append((source, target))
            if edge_attributes is not None:
                edge_attributes.append([_numeric_attribute(attrs, name) for name in attribute_fields])
            if add_reverse_edges:
                edges.append((target, source))
                if edge_attributes is not None:
                    edge_attributes.append([_numeric_attribute(attrs, name) for name in attribute_fields])

    edge_index, edge_attr = _unique_edges_with_attributes(edges, edge_attributes)
    if not bool(river_mask.any()):
        raise ValueError("Flowline builder did not intersect any active grid cells")

    metadata = {
        "flowline_file": str(file_path),
        "flowline_record_count": len(records),
        "flowline_coordinate_mode": resolved_mode,
        "flowline_attribute_edge_features": attribute_fields,
        "flowline_node_strategy": "river_cells",
        "flowline_direction": direction_mode,
        "flowline_densify_step": coordinate_step,
    }
    return edge_index, river_mask, edge_attr, attribute_fields, metadata


def _load_gauge_records(file_path: Path) -> list[dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(file_path, dtype=str)
        return df.to_dict(orient="records")
    if suffix == ".json":
        with file_path.open("r") as fp:
            payload = json.load(fp)
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, Mapping):
            if "gauges" in payload and isinstance(payload["gauges"], list):
                return [dict(item) for item in payload["gauges"]]
            return [dict(payload)]
    raise ValueError(f"Unsupported gauge file type: {file_path}")


def _resolve_gauge_identity(record: Mapping[str, Any], *, id_column: str | None = None) -> str | None:
    candidate_keys = [id_column] if id_column else []
    candidate_keys.extend(["gauge_id", "basin_id", "site_no", "id", "name"])
    for key in candidate_keys:
        if key and key in record and record[key] not in [None, ""]:
            return str(record[key])
    return None


def _canonical_gauge_id(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        return text
    if text.isdigit():
        stripped = text.lstrip("0")
        return stripped if stripped != "" else "0"
    return text


def _nearest_flat_index(
    lat: float,
    lon: float,
    *,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> int:
    distance = (lat2d - float(lat)) ** 2 + (lon2d - float(lon)) ** 2
    if active_mask is not None:
        masked_distance = np.where(active_mask, distance, np.inf)
        flat_idx = int(np.argmin(masked_distance))
        if not np.isfinite(masked_distance.reshape(-1)[flat_idx]):
            raise ValueError("No active routing node is available for nearest gauge mapping")
    else:
        flat_idx = int(np.argmin(distance))
    return flat_idx


def _nearest_xy_flat_index(
    y_value: float,
    x_value: float,
    *,
    y2d: np.ndarray,
    x2d: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> int:
    distance = (y2d - float(y_value)) ** 2 + (x2d - float(x_value)) ** 2
    if active_mask is not None:
        masked_distance = np.where(active_mask, distance, np.inf)
        flat_idx = int(np.argmin(masked_distance))
        if not np.isfinite(masked_distance.reshape(-1)[flat_idx]):
            raise ValueError("No active routing node is available for nearest projected-coordinate mapping")
    else:
        flat_idx = int(np.argmin(distance))
    return flat_idx


def _resolve_gauge_records(
    gauge_spec: Mapping[str, Any],
    *,
    grid_shape: Sequence[int],
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    y2d: np.ndarray | None = None,
    x2d: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
) -> tuple[list[int], list[str]]:
    items = gauge_spec.get("items", [])
    if items and not isinstance(items, list):
        raise ValueError("routing.graph.gauges.items must be a list when provided")
    records = [dict(item) for item in items]
    file_path = gauge_spec.get("file_path")
    if file_path:
        records.extend(_load_gauge_records(Path(file_path)))

    id_column = gauge_spec.get("id_column")
    y_column = gauge_spec.get("y_column")
    x_column = gauge_spec.get("x_column")
    lat_column = gauge_spec.get("lat_column")
    lon_column = gauge_spec.get("lon_column")
    coordinate_mode = str(gauge_spec.get("coordinate_mode", gauge_spec.get("coord_space", "grid"))).lower()

    gauge_indices: list[int] = []
    gauge_ids: list[str] = []
    for record in records:
        gauge_id = _resolve_gauge_identity(record, id_column=id_column)

        if "flat_index" in record:
            flat_index = int(record["flat_index"])
        elif "grid_index" in record:
            flat_index = int(record["grid_index"])
        elif (y_column and y_column in record and x_column and x_column in record) or (
            "y" in record and "x" in record
        ):
            raw_y = record[y_column] if y_column and y_column in record else record["y"]
            raw_x = record[x_column] if x_column and x_column in record else record["x"]
            if coordinate_mode in {"projected", "projection", "lambert", "map"}:
                if y2d is None or x2d is None:
                    raise ValueError("Projected x/y gauge mapping requires routing grid x2d/y2d coordinates")
                flat_index = _nearest_xy_flat_index(
                    float(raw_y),
                    float(raw_x),
                    y2d=y2d,
                    x2d=x2d,
                    active_mask=active_mask,
                )
            else:
                y_value = int(raw_y)
                x_value = int(raw_x)
                flat_index = _flatten_index(y_value, x_value, int(grid_shape[1]))
        elif (
            (lat_column and lat_column in record and lon_column and lon_column in record)
            or ("lat" in record and "lon" in record)
        ):
            if lat2d is None or lon2d is None:
                raise ValueError("lat2d/lon2d are required for gauge lat/lon mapping")
            lat_value = float(record[lat_column] if lat_column and lat_column in record else record["lat"])
            lon_value = float(record[lon_column] if lon_column and lon_column in record else record["lon"])
            flat_index = _nearest_flat_index(
                lat_value,
                lon_value,
                lat2d=lat2d,
                lon2d=lon2d,
                active_mask=active_mask,
            )
        else:
            raise ValueError(
                "Each gauge record must define one of: flat_index, grid_index, y/x, or lat/lon"
            )

        gauge_indices.append(flat_index)
        gauge_ids.append(gauge_id if gauge_id is not None else str(flat_index))

    return gauge_indices, gauge_ids


def resolve_gauge_mapping(
    gauge_spec: Mapping[str, Any] | None,
    *,
    grid_shape: Sequence[int],
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    y2d: np.ndarray | None = None,
    x2d: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    basin_ids: Sequence[str] | None = None,
) -> tuple[torch.Tensor | None, list[str]]:
    if not gauge_spec:
        return None, []

    num_nodes = int(grid_shape[0]) * int(grid_shape[1])
    if "gauge_index" in gauge_spec or "gauge_indices" in gauge_spec:
        values = gauge_spec.get("gauge_index", gauge_spec.get("gauge_indices"))
        gauge_index = torch.as_tensor(values, dtype=torch.long).reshape(-1)
        gauge_ids = [str(value) for value in gauge_spec.get("gauge_ids", [])]
    elif "gauge_mask" in gauge_spec:
        mask = torch.as_tensor(gauge_spec["gauge_mask"], dtype=torch.bool).reshape(-1)
        if mask.numel() != num_nodes:
            raise ValueError(
                f"gauge_mask length {mask.numel()} must match num_nodes {num_nodes}"
            )
        gauge_index = torch.nonzero(mask, as_tuple=False).reshape(-1)
        gauge_ids = [str(value) for value in gauge_spec.get("gauge_ids", [])]
    else:
        indices, gauge_ids = _resolve_gauge_records(
            gauge_spec,
            grid_shape=grid_shape,
            lat2d=lat2d,
            lon2d=lon2d,
            y2d=y2d,
            x2d=x2d,
            active_mask=active_mask,
        )
        gauge_index = torch.as_tensor(indices, dtype=torch.long).reshape(-1)

    if gauge_index.numel() == 0:
        return gauge_index, gauge_ids
    if int(gauge_index.max().item()) >= num_nodes or int(gauge_index.min().item()) < 0:
        raise ValueError("Gauge indices fall outside the routing grid")

    if basin_ids:
        basin_ids = [str(value) for value in basin_ids]
        if gauge_ids:
            index_by_id = {
                _canonical_gauge_id(gauge_id): int(idx)
                for gauge_id, idx in zip(gauge_ids, gauge_index.tolist())
            }
            missing = [basin_id for basin_id in basin_ids if _canonical_gauge_id(basin_id) not in index_by_id]
            if missing and len(basin_ids) == 1 and len(gauge_ids) == 1:
                gauge_ids = list(basin_ids)
            elif missing:
                raise ValueError(
                    "Gauge mapping is missing basin ids required by the dataset target order: "
                    f"{missing}"
                )
            if not missing:
                gauge_index = torch.as_tensor(
                    [index_by_id[_canonical_gauge_id(basin_id)] for basin_id in basin_ids],
                    dtype=torch.long,
                )
                gauge_ids = list(basin_ids)
        elif gauge_index.numel() == len(basin_ids):
            gauge_ids = list(basin_ids)

    return gauge_index, gauge_ids


def build_routing_graph_payload(
    *,
    builder: str,
    grid_shape: Sequence[int],
    mask_array: np.ndarray | None = None,
    mask_spec: Mapping[str, Any] | None = None,
    elevation_array: np.ndarray | None = None,
    flow_direction: np.ndarray | None = None,
    flow_direction_encoding: str = "arcgis",
    flowlines: Mapping[str, Any] | None = None,
    node_feature_array: np.ndarray | None = None,
    gauges: Mapping[str, Any] | None = None,
    derived_edge_features: Sequence[str] | None = None,
    edge_weight_feature: str | None = None,
    edge_weight_normalization: str | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    y2d: np.ndarray | None = None,
    x2d: np.ndarray | None = None,
    basin_ids: Sequence[str] | None = None,
    add_self_loops: bool = False,
    directed: bool | None = None,
    add_reverse_edges: bool | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    builder_name = str(builder).lower()
    y_size, x_size = int(grid_shape[0]), int(grid_shape[1])
    if mask_array is None:
        active_mask = np.ones((y_size, x_size), dtype=bool)
    else:
        active_mask = _resolve_active_mask(_as_2d_array("mask_array", mask_array), mask_spec)
    if active_mask.shape != (y_size, x_size):
        raise ValueError(
            f"Active mask shape {tuple(active_mask.shape)} must match grid_shape {(y_size, x_size)}"
        )

    precomputed_edge_attr = None
    precomputed_edge_feature_names: list[str] = []
    builder_metadata: dict[str, Any] = {}

    if builder_name == "grid_4_neighbor":
        if directed is None:
            directed = False
        if add_reverse_edges is None:
            add_reverse_edges = not directed
        edge_index = _build_grid_edges(
            grid_shape=grid_shape,
            active_mask=active_mask,
            diagonals=False,
            directed=bool(directed),
            add_reverse_edges=bool(add_reverse_edges),
            show_progress=show_progress,
        )
    elif builder_name == "grid_8_neighbor":
        if directed is None:
            directed = False
        if add_reverse_edges is None:
            add_reverse_edges = not directed
        edge_index = _build_grid_edges(
            grid_shape=grid_shape,
            active_mask=active_mask,
            diagonals=True,
            directed=bool(directed),
            add_reverse_edges=bool(add_reverse_edges),
            show_progress=show_progress,
        )
    elif builder_name == "flow_direction_d8":
        if flow_direction is None:
            raise ValueError("flow_direction_d8 builder requires a flow_direction array")
        if add_reverse_edges is None:
            add_reverse_edges = False
        edge_index = _build_flow_direction_edges(
            flow_direction=_as_2d_array("flow_direction", flow_direction),
            active_mask=active_mask,
            encoding=flow_direction_encoding,
            add_reverse_edges=bool(add_reverse_edges),
            show_progress=show_progress,
        )
    elif builder_name == "dem_downhill_d8":
        if elevation_array is None:
            elevation_array = mask_array
        if elevation_array is None:
            raise ValueError("dem_downhill_d8 builder requires an elevation array")
        if add_reverse_edges is None:
            add_reverse_edges = False
        edge_index = _build_dem_downhill_edges(
            elevation=_as_2d_array("elevation_array", elevation_array),
            active_mask=active_mask,
            add_reverse_edges=bool(add_reverse_edges),
            show_progress=show_progress,
        )
    elif builder_name in {"flowline_network", "river_network", "flowlines"}:
        if flowlines is None:
            raise ValueError("flowline_network builder requires routing.graph.flowlines")
        if add_reverse_edges is None:
            add_reverse_edges = False
        edge_index, active_mask, precomputed_edge_attr, precomputed_edge_feature_names, builder_metadata = (
            _build_flowline_network_edges(
                flowline_spec=flowlines,
                base_active_mask=active_mask,
                lat2d=lat2d,
                lon2d=lon2d,
                y2d=y2d,
                x2d=x2d,
                add_reverse_edges=bool(add_reverse_edges),
                show_progress=show_progress,
            )
        )
    else:
        raise ValueError(
            f"Unsupported routing graph builder '{builder}'. "
            "Choose from: grid_4_neighbor, grid_8_neighbor, flow_direction_d8, dem_downhill_d8, flowline_network."
        )

    gauge_index, gauge_ids = resolve_gauge_mapping(
        gauges,
        grid_shape=grid_shape,
        lat2d=lat2d,
        lon2d=lon2d,
        y2d=y2d,
        x2d=x2d,
        active_mask=active_mask,
        basin_ids=basin_ids,
    )

    edge_index, gauge_index, flat_index, node_y, node_x = _compact_graph_indices(
        edge_index=edge_index,
        gauge_index=gauge_index,
        active_mask=active_mask,
        x_size=x_size,
    )
    num_nodes = int(flat_index.numel())
    node_features = _compact_node_features(
        node_feature_array,
        active_mask=active_mask,
    )

    if add_self_loops:
        if precomputed_edge_attr is not None:
            raise ValueError("add_self_loops=true is not supported with flowline attribute edge features")
        self_loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_index = _unique_edge_index(list(map(tuple, edge_index.t().tolist())))

    edge_attr, edge_feature_names, edge_weight = _compute_compact_edge_features(
        edge_index=edge_index,
        flat_index=flat_index,
        x_size=x_size,
        elevation_array=_as_2d_array("elevation_array", elevation_array) if elevation_array is not None else None,
        y2d=_as_2d_array("y2d", y2d) if y2d is not None else None,
        x2d=_as_2d_array("x2d", x2d) if x2d is not None else None,
        feature_names=derived_edge_features,
        weight_feature=edge_weight_feature,
        weight_normalization=edge_weight_normalization,
    )
    if precomputed_edge_attr is not None:
        if edge_attr is None:
            edge_attr = precomputed_edge_attr
            edge_feature_names = list(precomputed_edge_feature_names)
        else:
            edge_attr = torch.cat([edge_attr, precomputed_edge_attr], dim=1)
            edge_feature_names = list(edge_feature_names) + list(precomputed_edge_feature_names)

    payload: dict[str, Any] = {
        "edge_index": edge_index,
        "flat_index": flat_index,
        "node_y": node_y,
        "node_x": node_x,
        "node_ids": [str(index) for index in flat_index.tolist()],
        "metadata": {
            "builder": builder_name,
            "grid_shape": (y_size, x_size),
            "full_grid_node_count": int(y_size * x_size),
            "active_node_count": int(active_mask.sum()),
            "flow_direction_encoding": flow_direction_encoding if builder_name == "flow_direction_d8" else None,
            "derived_edge_features": list(edge_feature_names),
            "edge_weight_feature": None if edge_weight_feature in {None, ""} else str(edge_weight_feature),
            "edge_weight_normalization": None
            if edge_weight_normalization in {None, ""}
            else str(edge_weight_normalization),
            **builder_metadata,
        },
    }
    if node_features is not None:
        payload["node_features"] = node_features
    if edge_attr is not None:
        payload["edge_attr"] = edge_attr
        payload["edge_feature_names"] = list(edge_feature_names)
    if edge_weight is not None:
        payload["edge_weight"] = edge_weight
    if gauge_index is not None:
        payload["gauge_index"] = gauge_index
        payload["gauge_ids"] = list(gauge_ids)
        gauge_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if gauge_index.numel():
            gauge_mask[gauge_index] = True
        payload["gauge_mask"] = gauge_mask
    return payload
