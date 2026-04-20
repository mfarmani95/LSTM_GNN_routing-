#!/usr/bin/env bash
set -euo pipefail

python -m lstm_gnn_routing.cli.main train \
  --config-file configs/lstm_gnn_ngen_curriculum.yml
