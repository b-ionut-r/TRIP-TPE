"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from trip_tpe.utils.config import (
    TRIPTPEConfig,
    TransformerConfig,
    TrainingConfig,
    load_config,
    save_config,
)


class TestConfig:
    """Test configuration loading, saving, and overrides."""

    def test_default_config(self):
        config = TRIPTPEConfig()
        assert config.transformer.num_layers == 4
        assert config.transformer.embed_dim == 256
        assert config.training.batch_size == 32
        assert config.region_proposal.n_warmup_trials == 5

    def test_save_and_load_config(self, tmp_path):
        config = TRIPTPEConfig()
        config.transformer.num_layers = 6
        config.training.learning_rate = 1e-3

        path = tmp_path / "test_config.yaml"
        save_config(config, path)
        loaded = load_config(path)

        assert loaded.transformer.num_layers == 6
        assert loaded.training.learning_rate == 1e-3
        assert loaded.transformer.embed_dim == 256  # Unchanged default

    def test_config_overrides(self):
        overrides = {
            "transformer": {"num_layers": 8},
            "training": {"batch_size": 64},
            "device": "cpu",
        }
        config = load_config(overrides=overrides)
        assert config.transformer.num_layers == 8
        assert config.training.batch_size == 64
        assert config.device == "cpu"

    def test_load_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        # Should return defaults
        assert config.transformer.num_layers == 4

    def test_tpe_config_defaults(self):
        config = TRIPTPEConfig()
        assert config.tpe.n_startup_trials == 0  # Transformer handles warmup
        assert config.tpe.multivariate is True
        assert config.tpe.gamma_value == 0.15  # Watanabe 2023 optimal
