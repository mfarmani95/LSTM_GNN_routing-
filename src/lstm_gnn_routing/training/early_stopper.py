from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Simple patience-based early stopping helper."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_value = float("inf")

    def reset(self) -> None:
        self.counter = 0
        self.best_value = float("inf")

    def step(self, current_value: float) -> bool:
        if current_value < self.best_value - self.min_delta:
            self.best_value = float(current_value)
            self.counter = 0
            return False

        self.counter += 1
        logger.info("Early stopping counter: %s/%s", self.counter, self.patience)
        return self.counter >= self.patience
