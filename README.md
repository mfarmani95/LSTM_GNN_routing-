# LSTM and GNN Routing

Standalone research code for testing a gridded ML runoff generator plus graph neural network routing model.

This repository is intentionally separated from the physics model code. It contains only the LSTM/temporal-convolution runoff models, graph-routing models, data loaders, preprocessing utilities, and training loop needed to test whether the routing architecture can learn streamflow from gridded meteorological inputs and a river/terrain graph.

## What This Repo Does

The workflow is:

1. Load daily gridded meteorological forcing from yearly Zarr stores.
2. Load static gridded fields such as `landmask`, `BOTSOIL2D`, `VEG2D`, or precomputed continuous fields.
3. Load observed daily streamflow CSVs for gauge targets.
4. Generate gridded runoff with a shared LSTM or temporal convolution model.
5. Optionally transfer grid-cell runoff to routing graph nodes using a sparse source-to-target mapping.
6. Route graph-node runoff to gauges with a GNN.
7. Train with masked losses so missing streamflow observations do not contribute to loss.

## Repository Layout

```text
configs/
  lstm_gnn_ngen_curriculum.yml     Example staged/curriculum training config
scripts/
  train_lstm_gnn.sh                Convenience training command
src/lstm_gnn_routing/
  cli/                             Command line entrypoint
  dataset/                         Dataset and DataLoader batching
  runoff_models/                   LSTM and temporal-convolution runoff models
  routing_models/                  GNN routing, runoff transfer, graph builders
  tools/                           Zarr aggregation, graph building, QC plotting, scaler computation
  training/                        Standalone trainer, losses, model factory
  utils/                           Config and data-loading helpers
```

## Data Expected

The example config assumes a local layout like this:

```text
data/
  aorc_daily_zarr/
    1981.zarr/
    1982.zarr/
    ...
  static/
    lon_lat.nc
    landmask.nc
    botsoil30s_res.nc
    veg30s_res.nc
  streamflow/
    26_basin_ids_ngen.txt
    daily/
      09489500.csv
      ...
  graphs/
    routing_graph_ngen_salt_verde_cache.nc
```

The graph cache should contain the directed routing graph plus graph/node metadata such as:

- `edge_index`
- `edge_weight`
- `node_features`
- `gauge_index`
- `gauge_ids`
- `runoff_source_flat_index`
- `runoff_target_index`
- `runoff_source_weight`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[preprocess]"
```

Install the PyTorch and PyTorch Geometric builds that match your CUDA environment before running on GPU.

## Train

```bash
python -m lstm_gnn_routing.cli.main train \
  --config-file configs/lstm_gnn_ngen_curriculum.yml
```

Or:

```bash
bash scripts/train_lstm_gnn.sh
```

## Compute Train-Period Scalers

The recommended workflow is to compute train-period scalers once, then reuse them for validation/test:

```bash
python -m lstm_gnn_routing.tools.compute_train_scaler \
  --config-file configs/lstm_gnn_ngen_curriculum.yml \
  --overwrite
```

The scaler file stores train-period statistics for normalized forcing, static inputs, and streamflow targets. Validation and test periods should load these same statistics instead of recomputing their own.

## Preprocessing Utilities

Hourly forcing can be converted to yearly Zarr stores:

```bash
python -m lstm_gnn_routing.tools.convert_hourly_forcing_to_zarr \
  --input-root /path/to/hourly/files \
  --output-root data/aorc_hourly_zarr \
  --mode yearly
```

Hourly Zarr forcing can be aggregated to daily forcing:

```bash
python -m lstm_gnn_routing.tools.aggregate_hourly_zarr_to_daily \
  --input-root data/aorc_hourly_zarr \
  --output-root data/aorc_daily_zarr
```

The daily aggregation handles precipitation as an accumulated depth and uses daily means for flux/state variables. Check the tool options before production conversion.

## Notes

This repo is meant for routing-model experiments. If you later want physics-model runoff, keep that workflow in a separate integration repository and pass cached or generated runoff into the routing model through the same `runoff_total` interface.
