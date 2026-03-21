"""Tests for trajectory dataset classes."""

import numpy as np
import pytest
import torch

from trip_tpe.data.preprocessing import TrajectoryPair, TrajectoryPreprocessor
from trip_tpe.data.trajectory_dataset import TrajectoryDataset


class TestTrajectoryDataset:
    """Test the PyTorch dataset for trajectory training pairs."""

    @pytest.fixture
    def sample_pairs(self):
        preprocessor = TrajectoryPreprocessor(seed=42)
        rng = np.random.RandomState(42)

        all_pairs = []
        for i in range(5):
            n_trials = rng.randint(30, 60)
            n_dims = 6
            configs = rng.rand(n_trials, n_dims).astype(np.float32)
            objectives = rng.rand(n_trials).astype(np.float32)
            pairs = preprocessor.process_trajectory(configs, objectives, f"test_{i}")
            all_pairs.extend(pairs)
        return all_pairs

    def test_dataset_length(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        assert len(dataset) == len(sample_pairs)

    def test_item_shapes(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        item = dataset[0]

        assert item["input_seq"].shape == (100, 17)  # max_seq_len x (hp_dim + 1)
        assert item["attention_mask"].shape == (100,)
        assert item["target_lower"].shape == (16,)
        assert item["target_upper"].shape == (16,)
        assert item["dim_mask"].shape == (16,)

    def test_attention_mask_valid(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        item = dataset[0]

        mask = item["attention_mask"]
        seq_len = item["seq_len"].item()
        assert mask[:seq_len].sum() == seq_len
        assert mask[seq_len:].sum() == 0

    def test_dim_mask_valid(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        item = dataset[0]
        dim_mask = item["dim_mask"]
        # First 6 dims should be active (n_dims=6), rest padded
        assert dim_mask[:6].sum() == 6
        assert dim_mask[6:].sum() == 0

    def test_target_bounds_valid(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        item = dataset[0]
        assert (item["target_lower"] <= item["target_upper"]).all()
        assert (item["target_lower"] >= 0).all()
        assert (item["target_upper"] <= 1).all()

    def test_collate_fn(self, sample_pairs):
        dataset = TrajectoryDataset(sample_pairs, max_seq_len=100, hp_dim=16)
        batch = TrajectoryDataset.collate_fn([dataset[0], dataset[1]])
        assert batch["input_seq"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2

    def test_dataloader_integration(self, sample_pairs):
        from torch.utils.data import DataLoader

        dataset = TrajectoryDataset(sample_pairs, max_seq_len=50, hp_dim=8)
        loader = DataLoader(dataset, batch_size=4, collate_fn=TrajectoryDataset.collate_fn)

        batch = next(iter(loader))
        assert batch["input_seq"].shape == (4, 50, 9)  # batch x seq x (hp_dim+1)
