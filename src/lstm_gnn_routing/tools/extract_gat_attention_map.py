"""Extract and plot mean GAT attention on the routing graph.

This tool runs a saved GAT routing model over an evaluation period and averages
the runtime attention coefficients for each original NGen routing edge.  The
checkpoint's ``att_src``/``att_dst``/``att_edge`` tensors are learned parameters;
this script records the actual per-edge attention coefficients produced during
model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from lstm_gnn_routing.dataset.batcher import RoutingBatcher
from lstm_gnn_routing.dataset.dataset import RoutingDataset
from lstm_gnn_routing.training.model_factory import (
    build_routing_model,
    build_runoff_model,
    build_runoff_transfer_model,
)
from lstm_gnn_routing.training.trainer import RunoffRoutingPipeline
from lstm_gnn_routing.utils.config import RoutingConfig


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _graph_edge_index(graph: Mapping[str, Any]) -> np.ndarray:
    edge_index = _to_numpy(graph["edge_index"]).astype(np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index with shape [2, E], got {edge_index.shape}")
    return edge_index


def _graph_xy(graph: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_keys = ("node_x", "x", "longitude", "lon")
    y_keys = ("node_y", "y", "latitude", "lat")
    x = next((_to_numpy(graph[key]).reshape(-1) for key in x_keys if key in graph), None)
    y = next((_to_numpy(graph[key]).reshape(-1) for key in y_keys if key in graph), None)
    return x, y


def _apply_coordinate_grid(
    graph: Mapping[str, Any],
    coordinate_grid_file: str | Path | None,
    *,
    plot_lon_lat: bool = False,
) -> Mapping[str, Any]:
    if coordinate_grid_file is None:
        return graph
    grid_path = _as_path(coordinate_grid_file)
    ds = xr.open_dataset(grid_path)
    if "x" not in ds.coords or "y" not in ds.coords:
        print(f"Coordinate grid {grid_path} has no x/y coordinates; keeping graph coordinates unchanged.")
        return graph
    if "node_x" not in graph or "node_y" not in graph:
        return graph

    node_x_idx = _to_numpy(graph["node_x"]).reshape(-1)
    node_y_idx = _to_numpy(graph["node_y"]).reshape(-1)
    if not np.all(np.isfinite(node_x_idx)) or not np.all(np.isfinite(node_y_idx)):
        return graph

    node_x_int = np.rint(node_x_idx).astype(np.int64)
    node_y_int = np.rint(node_y_idx).astype(np.int64)
    x_coord = np.asarray(ds.coords["x"].values)
    y_coord = np.asarray(ds.coords["y"].values)
    valid = (
        np.allclose(node_x_idx, node_x_int)
        and np.allclose(node_y_idx, node_y_int)
        and node_x_int.min(initial=0) >= 0
        and node_y_int.min(initial=0) >= 0
        and node_x_int.max(initial=-1) < x_coord.shape[0]
        and node_y_int.max(initial=-1) < y_coord.shape[0]
    )
    if not valid:
        print(f"Graph node_x/node_y do not look like indices for {grid_path}; keeping them unchanged.")
        return graph

    merged = dict(graph)
    merged["node_x_index"] = node_x_idx
    merged["node_y_index"] = node_y_idx
    merged["node_x"] = x_coord[node_x_int]
    merged["node_y"] = y_coord[node_y_int]
    if "spatial_ref" in ds:
        spatial_ref = ds["spatial_ref"].attrs
        merged["_plot_crs_wkt"] = spatial_ref.get("crs_wkt") or spatial_ref.get("spatial_ref")
    if plot_lon_lat and merged.get("_plot_crs_wkt") is not None:
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(merged["_plot_crs_wkt"], "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(merged["node_x"], merged["node_y"])
            merged["node_x"] = np.asarray(lon)
            merged["node_y"] = np.asarray(lat)
            merged["_plot_crs_wkt"] = "EPSG:4326"
            merged["_plot_lon_lat"] = True
        except Exception as exc:  # pragma: no cover - depends on optional pyproj/runtime CRS support
            print(f"Could not reproject graph coordinates to lon/lat; keeping projected coordinates ({exc}).")
    merged["_coordinate_grid_file"] = str(grid_path)
    return merged


def _edge_feature_frame(graph: Mapping[str, Any], edge_count: int) -> pd.DataFrame:
    edge_attr = None
    for key in ("edge_attr", "edge_features", "edge_feature"):
        if key in graph:
            edge_attr = _to_numpy(graph[key])
            break
    if edge_attr is None:
        return pd.DataFrame({"edge_id": np.arange(edge_count, dtype=np.int64)})
    edge_attr = np.asarray(edge_attr)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr[:, None]
    if edge_attr.shape[0] != edge_count and edge_attr.shape[1] == edge_count:
        edge_attr = edge_attr.T
    names = graph.get("edge_feature_names") or graph.get("edge_attr_names")
    if names is None:
        metadata = graph.get("metadata", {}) or {}
        names = metadata.get("edge_feature_names") or metadata.get("edge_features")
    if names is None:
        names = [f"edge_feature_{idx}" for idx in range(edge_attr.shape[1])]
    names = [str(name) for name in list(names)]
    if len(names) != edge_attr.shape[1]:
        names = [f"edge_feature_{idx}" for idx in range(edge_attr.shape[1])]
    frame = pd.DataFrame(edge_attr, columns=names)
    frame.insert(0, "edge_id", np.arange(edge_count, dtype=np.int64))
    return frame


def _node_feature_frame(graph: Mapping[str, Any]) -> pd.DataFrame:
    node_features = None
    for key in ("node_features", "node_attr", "node_feature"):
        if key in graph:
            node_features = _to_numpy(graph[key])
            break
    if node_features is None:
        return pd.DataFrame()
    node_features = np.asarray(node_features)
    if node_features.ndim == 1:
        node_features = node_features[:, None]
    names = graph.get("node_feature_names") or graph.get("node_attr_names")
    if names is None:
        metadata = graph.get("metadata", {}) or {}
        names = metadata.get("node_feature_names") or metadata.get("node_features")
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except json.JSONDecodeError:
            names = [name.strip() for name in names.split(",")]
    if names is None or len(names) != node_features.shape[1]:
        names = [f"node_feature_{idx}" for idx in range(node_features.shape[1])]
    frame = pd.DataFrame(node_features, columns=[str(name) for name in names])
    frame.insert(0, "node_id", np.arange(node_features.shape[0], dtype=np.int64))
    return frame


def _upstream_edges_for_gauge(edge_index: np.ndarray, gauge_node: int) -> np.ndarray:
    incoming: dict[int, list[int]] = {}
    for edge_id, target in enumerate(edge_index[1]):
        incoming.setdefault(int(target), []).append(edge_id)
    visited_nodes = {int(gauge_node)}
    upstream_edges: set[int] = set()
    stack = [int(gauge_node)]
    while stack:
        node = stack.pop()
        for edge_id in incoming.get(node, []):
            if edge_id in upstream_edges:
                continue
            upstream_edges.add(edge_id)
            source = int(edge_index[0, edge_id])
            if source not in visited_nodes:
                visited_nodes.add(source)
                stack.append(source)
    return np.asarray(sorted(upstream_edges), dtype=np.int64)


def _safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan
    denom = float(np.sum(weights[mask]))
    if abs(denom) < 1e-12:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / denom)


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.mean(np.sort(values)[-count:]))


def _save_gauge_upstream_metrics(
    output_dir: Path,
    graph: Mapping[str, Any],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    if "gauge_index" not in graph:
        print("Graph has no gauge_index; skipped gauge upstream attention metrics.")
        return pd.DataFrame()

    edge_index = summary[["from_node", "to_node"]].to_numpy(dtype=np.int64).T
    gauge_index = _to_numpy(graph["gauge_index"]).reshape(-1).astype(np.int64)
    gauge_ids = graph.get("gauge_ids") or graph.get("gauge_id")
    if gauge_ids is None:
        gauge_ids = [str(idx) for idx in range(len(gauge_index))]
    else:
        gauge_ids = [str(value) for value in _to_numpy(gauge_ids).reshape(-1).tolist()]

    attention = summary["attention_mean"].to_numpy(dtype=float)
    indegree = np.bincount(edge_index[1], minlength=int(edge_index.max()) + 1)
    node_frame = _node_feature_frame(graph)
    stream_order = None
    drainage_area = None
    if not node_frame.empty:
        if "order" in node_frame:
            stream_order = node_frame["order"].to_numpy(dtype=float)
        for candidate in ("upstream_area_km2_topologic", "tot_drainage_areasqkm", "areasqkm"):
            if candidate in node_frame:
                drainage_area = node_frame[candidate].to_numpy(dtype=float)
                break

    rows: list[dict[str, float | int | str]] = []
    for gauge_id, gauge_node in zip(gauge_ids, gauge_index):
        upstream_edge_ids = _upstream_edges_for_gauge(edge_index, int(gauge_node))
        row: dict[str, float | int | str] = {
            "gauge_id": gauge_id,
            "gauge_node": int(gauge_node),
            "upstream_edge_count": int(upstream_edge_ids.size),
        }
        if upstream_edge_ids.size == 0:
            rows.append(row)
            continue

        att = attention[upstream_edge_ids]
        row["mean_upstream_attention"] = float(np.nanmean(att))
        row["median_upstream_attention"] = float(np.nanmedian(att))
        row["top10pct_upstream_attention"] = _top_fraction_mean(att, 0.10)

        downstream_nodes = edge_index[1, upstream_edge_ids]
        confluence_mask = indegree[downstream_nodes] >= 2
        row["confluence_edge_count"] = int(np.sum(confluence_mask))
        row["attention_near_confluences"] = (
            float(np.nanmean(att[confluence_mask])) if np.any(confluence_mask) else np.nan
        )

        mainstem_mask = np.zeros_like(att, dtype=bool)
        if stream_order is not None:
            downstream_order = stream_order[downstream_nodes]
            if np.any(np.isfinite(downstream_order)):
                max_order = np.nanmax(downstream_order)
                mainstem_mask = downstream_order >= max_order
                row["mainstem_definition"] = "downstream_node_max_stream_order"
                row["mainstem_max_order"] = float(max_order)
        if not np.any(mainstem_mask) and drainage_area is not None:
            downstream_area = drainage_area[downstream_nodes]
            if np.any(np.isfinite(downstream_area)):
                threshold = float(np.nanquantile(downstream_area, 0.80))
                mainstem_mask = downstream_area >= threshold
                row["mainstem_definition"] = "downstream_node_top20pct_drainage_area"
                row["mainstem_area_threshold"] = threshold
        row["mainstem_edge_count"] = int(np.sum(mainstem_mask))
        row["attention_on_mainstem_edges"] = (
            float(np.nanmean(att[mainstem_mask])) if np.any(mainstem_mask) else np.nan
        )

        for attr in ("travel_time_proxy", "So", "Length_m", "n", "BtmWdth", "TopWdth"):
            if attr in summary:
                values = summary[attr].to_numpy(dtype=float)[upstream_edge_ids]
                row[f"mean_upstream_{attr}"] = float(np.nanmean(values))
                row[f"attention_weighted_{attr}"] = _safe_weighted_mean(values, att)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "gauge_upstream_attention_metrics.csv", index=False)
    return frame


class GATAttentionCollector:
    """Temporarily wraps GATConv layers and accumulates original-edge attention."""

    def __init__(self, routing_model: torch.nn.Module, edge_count: int):
        self.routing_model = routing_model
        self.edge_count = int(edge_count)
        self._original_forwards: list[tuple[torch.nn.Module, Any]] = []
        self._sum_by_layer: Dict[int, torch.Tensor] = {}
        self._count_by_layer: Dict[int, int] = {}

    def __enter__(self) -> "GATAttentionCollector":
        convs = getattr(self.routing_model, "convs", None)
        if convs is None:
            raise ValueError("Routing model has no 'convs' attribute; cannot collect GAT attention.")
        for layer_idx, conv in enumerate(convs):
            if not all(hasattr(conv, name) for name in ("att_src", "att_dst")):
                continue
            original_forward = conv.forward
            self._original_forwards.append((conv, original_forward))

            def wrapped_forward(x, edge_index, *args, _layer_idx=layer_idx, _orig=original_forward, **kwargs):
                kwargs["return_attention_weights"] = True
                output = _orig(x, edge_index, *args, **kwargs)
                node_output, attention = output
                returned_edge_index, alpha = attention
                self._accumulate(_layer_idx, edge_index, returned_edge_index, alpha)
                return node_output

            conv.forward = wrapped_forward  # type: ignore[method-assign]
        if not self._original_forwards:
            raise ValueError("No GAT-like convolution layers were found in routing_model.convs.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for conv, original_forward in self._original_forwards:
            conv.forward = original_forward  # type: ignore[method-assign]

    def _accumulate(
        self,
        layer_idx: int,
        input_edge_index: torch.Tensor,
        returned_edge_index: torch.Tensor,
        alpha: torch.Tensor,
    ) -> None:
        del returned_edge_index
        repeated_edge_count = int(input_edge_index.shape[1])
        if repeated_edge_count % self.edge_count != 0:
            raise ValueError(
                f"Repeated edge count {repeated_edge_count} is not divisible by graph edge count {self.edge_count}."
            )
        graph_count = repeated_edge_count // self.edge_count
        original_alpha = alpha[:repeated_edge_count].detach().float().cpu()
        if original_alpha.ndim == 1:
            original_alpha = original_alpha[:, None]
        original_alpha = original_alpha.reshape(graph_count, self.edge_count, original_alpha.shape[-1])
        layer_sum = original_alpha.sum(dim=0)
        if layer_idx not in self._sum_by_layer:
            self._sum_by_layer[layer_idx] = torch.zeros_like(layer_sum)
            self._count_by_layer[layer_idx] = 0
        self._sum_by_layer[layer_idx] += layer_sum
        self._count_by_layer[layer_idx] += graph_count

    def layer_means(self) -> Dict[int, np.ndarray]:
        means: Dict[int, np.ndarray] = {}
        for layer_idx, layer_sum in self._sum_by_layer.items():
            count = max(1, self._count_by_layer[layer_idx])
            means[layer_idx] = (layer_sum / count).numpy()
        return means


def _build_pipeline(config: RoutingConfig, example_batch: Mapping[str, Any], device: torch.device) -> RunoffRoutingPipeline:
    runoff_model = build_runoff_model(config, example_batch=example_batch, device=device)
    runoff_transfer = build_runoff_transfer_model(config, example_batch=example_batch, device=device)
    routing_model = build_routing_model(config, example_batch=example_batch, device=device)
    model = RunoffRoutingPipeline(
        runoff_model=runoff_model,
        runoff_transfer=runoff_transfer,
        routing_model=routing_model,
        runoff_device=device,
        routing_device=device,
    )
    return model.to(device)


def _save_attention_tables(
    output_dir: Path,
    graph: Mapping[str, Any],
    layer_means: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    edge_index = _graph_edge_index(graph)
    edge_count = edge_index.shape[1]
    summary = pd.DataFrame(
        {
            "edge_id": np.arange(edge_count, dtype=np.int64),
            "from_node": edge_index[0],
            "to_node": edge_index[1],
        }
    )
    head_rows: list[dict[str, float | int]] = []
    layer_columns = []
    for layer_idx in sorted(layer_means):
        mean_by_head = layer_means[layer_idx]
        layer_mean = mean_by_head.mean(axis=1)
        col = f"attention_layer_{layer_idx}"
        summary[col] = layer_mean
        layer_columns.append(col)
        for edge_id in range(edge_count):
            row: dict[str, float | int] = {"edge_id": edge_id, "layer": layer_idx}
            for head_idx in range(mean_by_head.shape[1]):
                row[f"head_{head_idx}"] = float(mean_by_head[edge_id, head_idx])
            row["head_mean"] = float(layer_mean[edge_id])
            head_rows.append(row)
    summary["attention_mean"] = summary[layer_columns].mean(axis=1)
    summary = summary.merge(_edge_feature_frame(graph, edge_count), on="edge_id", how="left")
    summary.to_csv(output_dir / "edge_attention_summary.csv", index=False)
    pd.DataFrame(head_rows).to_csv(output_dir / "edge_attention_by_layer_head.csv", index=False)
    return summary


def _plot_boundary(ax: plt.Axes, boundary_file: str | Path | None, target_crs: Any) -> None:
    if boundary_file is None:
        return
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        print(f"Could not import geopandas; skipped basin boundary overlay ({exc}).")
        return
    boundary_path = _as_path(boundary_file)
    gdf = gpd.read_file(boundary_path)
    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, alpha=0.9, zorder=5)


def _network_segments(graph: Mapping[str, Any], summary: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    x, y = _graph_xy(graph)
    if x is None or y is None:
        return None, None
    edge_index = summary[["from_node", "to_node"]].to_numpy(dtype=np.int64).T
    segments = np.stack(
        [
            np.column_stack([x[edge_index[0]], y[edge_index[0]]]),
            np.column_stack([x[edge_index[1]], y[edge_index[1]]]),
        ],
        axis=1,
    )
    return segments, edge_index


def _plot_network_attention(
    output_dir: Path,
    graph: Mapping[str, Any],
    summary: pd.DataFrame,
    *,
    boundary_file: str | Path | None = None,
) -> None:
    segments, _ = _network_segments(graph, summary)
    if segments is None:
        print("Graph has no node coordinates; skipped network PNG.")
        return
    values = summary["attention_mean"].to_numpy(dtype=float)
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    ):
        fig, ax = plt.subplots(figsize=(9.6, 7.4), constrained_layout=True)
        collection = LineCollection(segments, array=values, cmap="jet", linewidths=1.6, alpha=0.95)
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title("a) Mean GAT Attention on NGen River Edges")
        if graph.get("_plot_lon_lat"):
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        elif graph.get("_plot_crs_wkt") is not None:
            ax.set_xlabel("Projected x (m)")
            ax.set_ylabel("Projected y (m)")
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        _plot_boundary(ax, boundary_file, graph.get("_plot_crs_wkt"))
        cbar = fig.colorbar(collection, ax=ax, shrink=0.82)
        cbar.set_label("Mean attention coefficient", fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        fig.savefig(output_dir / "river_network_mean_gat_attention.png", dpi=300)
        plt.close(fig)


def _plot_layerwise_network_attention(
    output_dir: Path,
    graph: Mapping[str, Any],
    summary: pd.DataFrame,
    *,
    boundary_file: str | Path | None = None,
) -> None:
    segments, _ = _network_segments(graph, summary)
    if segments is None:
        print("Graph has no node coordinates; skipped layer-wise network PNG.")
        return
    layer_cols = [col for col in summary.columns if col.startswith("attention_layer_")]
    if not layer_cols:
        return
    values = summary[layer_cols].to_numpy(dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    ncols = min(2, len(layer_cols))
    nrows = int(np.ceil(len(layer_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    mappable = None
    for ax, col in zip(axes_arr, layer_cols):
        collection = LineCollection(
            segments,
            array=summary[col].to_numpy(dtype=float),
            cmap="viridis",
            linewidths=1.1,
            alpha=0.95,
            clim=(vmin, vmax),
        )
        ax.add_collection(collection)
        mappable = collection
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        layer_name = col.replace("attention_layer_", "GAT layer ")
        ax.set_title(layer_name)
        ax.set_xticks([])
        ax.set_yticks([])
        _plot_boundary(ax, boundary_file, graph.get("_plot_crs_wkt"))
    for ax in axes_arr[len(layer_cols) :]:
        ax.axis("off")
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes_arr[: len(layer_cols)], shrink=0.86)
        cbar.set_label("Mean attention coefficient")
    fig.suptitle("Layer-Wise Mean GAT Attention on NGen River Edges")
    fig.savefig(output_dir / "river_network_gat_attention_by_layer.png", dpi=300)
    plt.close(fig)


def _binned_median(x: np.ndarray, y: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.asarray([]), np.asarray([])
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return np.asarray([]), np.asarray([])
    centers: list[float] = []
    medians: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x >= left) & (x <= right if right == edges[-1] else x < right)
        if not np.any(in_bin):
            continue
        centers.append(float(np.nanmedian(x[in_bin])))
        medians.append(float(np.nanmedian(y[in_bin])))
    return np.asarray(centers), np.asarray(medians)


def _plot_linear_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, log_x: bool) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if log_x:
        finite &= x > 0
    if np.sum(finite) < 3:
        return
    x_fit = np.log10(x[finite]) if log_x else x[finite]
    y_fit = y[finite]
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    x_line = np.linspace(np.nanmin(x_fit), np.nanmax(x_fit), 100)
    y_line = slope * x_line + intercept
    plot_x = 10.0**x_line if log_x else x_line
    ax.plot(plot_x, y_line, color="#b12a1c", linewidth=3.0)


def _pretty_feature_label(name: str) -> str:
    labels = {
        "Length_m": "Reach length",
        "So": "Channel slope",
        "n": "Manning n",
        "BtmWdth": "Bottom width",
        "TopWdth": "Top width",
        "travel_time_proxy": "Travel-time proxy",
    }
    return labels.get(name, name)


def _plot_physical_controls_panel(output_dir: Path, summary: pd.DataFrame) -> None:
    preferred = ["Length_m", "So", "n", "BtmWdth", "TopWdth", "travel_time_proxy"]
    cols = [col for col in preferred if col in summary.columns and pd.api.types.is_numeric_dtype(summary[col])]
    if not cols:
        return
    ncols = min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.9 * ncols, 4.6 * nrows), constrained_layout=True)
        axes_arr = np.atleast_1d(axes).reshape(-1)
        y = summary["attention_mean"].to_numpy(dtype=float)
        for ax, col in zip(axes_arr, cols):
            x = summary[col].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            log_x = col in {"Length_m", "travel_time_proxy"} and np.any(finite) and np.nanmin(x[finite]) > 0
            ax.scatter(x[finite], y[finite], s=12, alpha=0.24, color="#2f6f8f", edgecolors="none")
            _plot_linear_trend(ax, x, y, log_x=log_x)
            rho = pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman") if np.sum(finite) > 2 else np.nan
            label = _pretty_feature_label(col)
            ax.set_title(rf"{label} ($\rho$={rho:.2f})")
            ax.set_xlabel(label)
            ax.set_ylabel("Mean GAT attention")
            if log_x:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.22, linewidth=0.6)
        for ax in axes_arr[len(cols) :]:
            ax.axis("off")
        fig.suptitle("b) GAT Attention vs Physical Routing Controls", fontsize=21)
        fig.savefig(output_dir / "gat_attention_physical_controls.png", dpi=300)
        plt.close(fig)


def _plot_stream_order_boxplot(output_dir: Path, graph: Mapping[str, Any], summary: pd.DataFrame) -> None:
    node_frame = _node_feature_frame(graph)
    if node_frame.empty or "order" not in node_frame:
        print("Graph has no node stream-order feature; skipped stream-order boxplot.")
        return
    edge_order = summary[["edge_id", "from_node", "to_node", "attention_mean"]].copy()
    node_order = node_frame.set_index("node_id")["order"]
    edge_order["downstream_stream_order"] = edge_order["to_node"].map(node_order).round().astype("Int64")
    edge_order = edge_order.dropna(subset=["downstream_stream_order", "attention_mean"]).copy()
    edge_order["downstream_stream_order"] = edge_order["downstream_stream_order"].astype(int)
    edge_order = edge_order[edge_order["downstream_stream_order"] > 0]
    if edge_order.empty:
        return
    edge_order.to_csv(output_dir / "gat_attention_by_stream_order.csv", index=False)

    orders = sorted(edge_order["downstream_stream_order"].unique())
    data = [
        edge_order.loc[edge_order["downstream_stream_order"] == order, "attention_mean"].to_numpy()
        for order in orders
    ]
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
        box = ax.boxplot(
            data,
            tick_labels=[str(order) for order in orders],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#7a1f12", "linewidth": 2.0},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        colors = plt.cm.YlGnBu(np.linspace(0.35, 0.85, len(orders)))
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        rng = np.random.default_rng(42)
        for i, values in enumerate(data, start=1):
            sample = values if len(values) <= 450 else rng.choice(values, size=450, replace=False)
            ax.scatter(
                rng.normal(i, 0.045, size=len(sample)),
                sample,
                s=7,
                alpha=0.18,
                color="0.18",
                edgecolors="none",
            )
        ax.set_xlabel("Downstream node stream order")
        ax.set_ylabel("Mean GAT attention coefficient")
        ax.set_title("c) Mean GAT Attention Grouped by Stream Order")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        fig.savefig(output_dir / "gat_attention_by_stream_order_boxplot.png", dpi=300)
        plt.close(fig)


def _plot_compound_attention_figure(output_dir: Path) -> None:
    panel_paths = {
        "a": output_dir / "river_network_mean_gat_attention.png",
        "b": output_dir / "gat_attention_physical_controls.png",
        "c": output_dir / "gat_attention_by_stream_order_boxplot.png",
    }
    if not all(path.exists() for path in panel_paths.values()):
        missing = [str(path.name) for path in panel_paths.values() if not path.exists()]
        print(f"Skipped compound attention figure because these panels are missing: {missing}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 20.5), constrained_layout=True)
    for label, ax in zip(("a", "b", "c"), axes):
        ax.imshow(plt.imread(panel_paths[label]))
        ax.axis("off")
    fig.savefig(output_dir / "gat_attention_compound_abc.png", dpi=300)
    plt.close(fig)


def _plot_attribute_scatter(output_dir: Path, summary: pd.DataFrame) -> None:
    skip = {"edge_id", "from_node", "to_node", "attention_mean"}
    skip.update(col for col in summary.columns if col.startswith("attention_layer_"))
    numeric_cols = [
        col
        for col in summary.columns
        if col not in skip and pd.api.types.is_numeric_dtype(summary[col])
    ]
    if not numeric_cols:
        return
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
        ax.scatter(summary[col], summary["attention_mean"], s=10, alpha=0.55)
        ax.set_xlabel(col)
        ax.set_ylabel("Mean GAT attention")
        ax.set_title(f"Attention vs {col}")
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in col)
        fig.savefig(output_dir / f"attention_vs_{safe}.png", dpi=220)
        plt.close(fig)


def _merge_plot_graph_file(graph: Mapping[str, Any], graph_file: str | Path | None) -> Mapping[str, Any]:
    if graph_file is None:
        return graph
    graph_path = _as_path(graph_file)
    ds = xr.open_dataset(graph_path)
    merged = dict(graph)
    if "edge_index" not in merged and {"edge_source", "edge_target"}.issubset(ds.variables):
        merged["edge_index"] = np.stack(
            [
                ds["edge_source"].values.astype(np.int64),
                ds["edge_target"].values.astype(np.int64),
            ],
            axis=0,
        )
    for key in ("node_x", "node_y", "edge_weight"):
        if key in ds.variables and key not in merged:
            merged[key] = ds[key].values
    # Prefer raw NetCDF edge attributes for interpretation/plots.  The dataset
    # graph can contain transformed model inputs, which are correct for inference
    # but less useful for paper-facing physical diagnostics.
    if "edge_attr" in ds.variables:
        merged["edge_attr"] = ds["edge_attr"].values
    if "edge_feature" in ds.coords:
        merged["edge_feature_names"] = [str(value) for value in ds["edge_feature"].values.tolist()]
    if "node_features" in ds.variables:
        merged["node_features"] = ds["node_features"].values
    if "node_feature" in ds.coords:
        merged["node_feature_names"] = [str(value) for value in ds["node_feature"].values.tolist()]
    for key in ("gauge_index", "gauge_id"):
        if key in ds.variables:
            merged[key] = ds[key].values
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Saved training run directory containing config.yml.")
    parser.add_argument("--checkpoint", default="best_final_stage_model.pt", help="Checkpoint filename or path.")
    parser.add_argument("--period", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Optional routing graph NetCDF used to add node_x/node_y coordinates for plotting.",
    )
    parser.add_argument(
        "--boundary-file",
        default=None,
        help="Optional basin boundary vector file, e.g. data/HUC4/Salt_and_Verde.shp.",
    )
    parser.add_argument(
        "--coordinate-grid-file",
        default=None,
        help="Optional grid NetCDF with x/y coordinates used to convert graph node_x/node_y indices to map coordinates.",
    )
    parser.add_argument(
        "--plot-lon-lat",
        action="store_true",
        help="Reproject coordinate-grid x/y coordinates to longitude/latitude for map plots.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional quick-look limit before full test extraction.")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _as_path(args.run_dir)
    config_path = run_dir / "config.yml"
    checkpoint_path = _as_path(args.checkpoint) if Path(args.checkpoint).is_absolute() else run_dir / args.checkpoint
    output_dir = _as_path(args.output_dir) if args.output_dir else run_dir / f"attention_{args.period}_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    config = RoutingConfig.from_yaml(config_path)
    dataset = RoutingDataset(config, period=args.period)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=RoutingBatcher.collate_fn,
    )
    example_batch = RoutingBatcher.collate_fn([dataset[0]])
    model = _build_pipeline(config, _move_to_device(example_batch, device), device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph = dataset.routing_graph
    if graph is None:
        graph = example_batch["routing_graph"]
    graph = _merge_plot_graph_file(graph, args.graph_file)
    graph = _apply_coordinate_grid(graph, args.coordinate_grid_file, plot_lon_lat=args.plot_lon_lat)
    edge_count = _graph_edge_index(graph).shape[1]

    with torch.no_grad(), GATAttentionCollector(model.routing_model, edge_count=edge_count) as collector:
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = _move_to_device(batch, device)
            _ = model(batch)
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1} evaluation batches...")

    layer_means = collector.layer_means()
    if not layer_means:
        raise RuntimeError("No attention values were collected.")
    summary = _save_attention_tables(output_dir, graph, layer_means)
    _save_gauge_upstream_metrics(output_dir, graph, summary)
    _plot_network_attention(output_dir, graph, summary, boundary_file=args.boundary_file)
    _plot_layerwise_network_attention(output_dir, graph, summary, boundary_file=args.boundary_file)
    _plot_physical_controls_panel(output_dir, summary)
    _plot_stream_order_boxplot(output_dir, graph, summary)
    _plot_compound_attention_figure(output_dir)
    _plot_attribute_scatter(output_dir, summary)

    manifest_path = output_dir / "attention_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_dir", str(run_dir)])
        writer.writerow(["checkpoint", str(checkpoint_path)])
        writer.writerow(["period", args.period])
        writer.writerow(["edge_count", edge_count])
        writer.writerow(["layers", ",".join(str(idx) for idx in sorted(layer_means))])
        writer.writerow(["max_batches", args.max_batches if args.max_batches is not None else "all"])
    print(f"Saved GAT attention artifacts to {output_dir}")


if __name__ == "__main__":
    main()
