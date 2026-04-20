from __future__ import annotations

from lstm_gnn_routing.training.trainer import LSTMGNNTrainer
from lstm_gnn_routing.utils.config import RoutingConfig


def start_training(config: RoutingConfig) -> LSTMGNNTrainer:
    trainer = LSTMGNNTrainer(config)
    trainer.train_and_validate()
    return trainer
