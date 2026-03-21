"""Tests for trajectory preprocessing pipeline."""

import numpy as np
import pytest

from trip_tpe.data.preprocessing import TrajectoryPreprocessor, TrajectoryPair


class TestTrajectoryPreprocessor:
    """Test the trajectory preprocessing pipeline."""

    @pytest.fixture
    def preprocessor(self):
        return TrajectoryPreprocessor(
            gamma=0.15,
            min_prefix_len=5,
            max_prefix_len=30,
            n_prefixes_per_trajectory=3,
            seed=42,
        )

    @pytest.fixture
    def sample_trajectory(self):
        rng = np.random.RandomState(42)
        n_trials = 50
        n_dims = 5
        configs = rng.uniform(0, 1, size=(n_trials, n_dims)).astype(np.float32)
        center = np.array([0.3, 0.5, 0.7, 0.2, 0.8])
        objectives = np.sum((configs - center)**2, axis=1).astype(np.float32)
        return configs, objectives

    def test_normalize_objectives(self, preprocessor):
        objectives = np.array([10.0, 5.0, 1.0, 20.0, 15.0])
        normalized = preprocessor.normalize_objectives(objectives)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        # Best value (1.0) should have lowest rank
        assert normalized[2] == 0.0
        # Worst value (20.0) should have highest rank
        assert normalized[3] == 1.0

    def test_compute_target_region(self, preprocessor, sample_trajectory):
        configs, objectives = sample_trajectory
        lower, upper = preprocessor.compute_target_region(configs, objectives, minimize=True)

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert np.all(lower >= 0.0)
        assert np.all(upper <= 1.0)
        assert np.all(lower < upper)

    def test_process_trajectory(self, preprocessor, sample_trajectory):
        configs, objectives = sample_trajectory
        pairs = preprocessor.process_trajectory(
            configs, objectives, search_space_id="test", minimize=True
        )

        assert len(pairs) > 0
        for pair in pairs:
            assert isinstance(pair, TrajectoryPair)
            assert pair.input_configs.shape[1] == 5  # n_dims
            assert pair.input_objectives.shape[1] == 1
            assert pair.target_lower.shape == (5,)
            assert pair.target_upper.shape == (5,)
            assert pair.seq_len >= 5  # min_prefix_len
            assert np.all(pair.target_lower < pair.target_upper)

    def test_short_trajectory_skipped(self, preprocessor):
        configs = np.random.rand(3, 4).astype(np.float32)
        objectives = np.random.rand(3).astype(np.float32)
        pairs = preprocessor.process_trajectory(configs, objectives)
        assert len(pairs) == 0  # Too short

    def test_process_batch(self, preprocessor):
        trajectories = []
        for i in range(5):
            rng = np.random.RandomState(i)
            n = rng.randint(30, 60)
            d = rng.randint(3, 8)
            trajectories.append({
                "configs": rng.rand(n, d).astype(np.float32),
                "objectives": rng.rand(n).astype(np.float32),
                "search_space_id": f"batch_{i}",
            })

        pairs = preprocessor.process_batch(trajectories)
        assert len(pairs) > 0
