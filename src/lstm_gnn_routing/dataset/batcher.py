from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from lstm_gnn_routing.dataset.dataset import RoutingDataset


class RoutingBatcher:
    """Simple wrapper that creates DataLoaders for routing datasets."""

    def __init__(self, train_dataset: RoutingDataset, val_dataset: RoutingDataset = None, test_dataset: RoutingDataset = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def collate_fn(batch: List[Dict]):
        forcing_lengths = {int(sample["x_forcing_physical"].shape[0]) for sample in batch}
        target_lengths = {int(sample["y"].shape[0]) for sample in batch}
        if len(forcing_lengths) > 1 or len(target_lengths) > 1:
            raise ValueError(
                "Batch contains samples with different sequence lengths. "
                "This can happen with calendar-based windows across leap years. "
                "Use batch_size=1 or a custom padded batching strategy."
            )

        collated = {
            "x_forcing_physical": torch.stack([sample["x_forcing_physical"] for sample in batch], dim=0),
            "x_static_physical": torch.stack([sample["x_static_physical"] for sample in batch], dim=0),
            "y": torch.stack([sample["y"] for sample in batch], dim=0),
            "target_mask": torch.stack([sample["target_mask"] for sample in batch], dim=0),
            "loss_mask": torch.stack([sample["loss_mask"] for sample in batch], dim=0),
            "spinup_mask": torch.stack([sample["spinup_mask"] for sample in batch], dim=0),
            "prediction_context_steps": torch.stack([sample["prediction_context_steps"] for sample in batch], dim=0),
            "x_info": [sample["x_info"] for sample in batch],
        }
        for key in ("routing_context_steps", "runoff_context_steps", "runoff_pre_routing_trim_steps"):
            if key in batch[0]:
                collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

        optional_tensor_keys = (
            "x_forcing_ml",
            "x_static_ml",
            "x_evolving_ml",
            "x_routing_static",
            "x_routing_dynamic",
        )
        for key in optional_tensor_keys:
            if key in batch[0]:
                collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

        if "routing_graph" in batch[0]:
            collated["routing_graph"] = batch[0]["routing_graph"]

        return collated

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataset.config.batch_size,
            shuffle=True,
            num_workers=self.train_dataset.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def validation_loader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataset.config.batch_size,
            shuffle=False,
            num_workers=self.val_dataset.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset is not set.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataset.config.batch_size,
            shuffle=False,
            num_workers=self.test_dataset.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
