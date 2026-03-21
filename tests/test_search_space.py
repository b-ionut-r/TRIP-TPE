"""Tests for search space encoding/decoding utilities."""

import numpy as np
import pytest

try:
    import optuna
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from trip_tpe.utils.search_space import SearchSpaceEncoder, apply_region_bounds


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestSearchSpaceEncoder:
    """Test search space encoding and decoding."""

    @pytest.fixture
    def mixed_space(self):
        return {
            "lr": FloatDistribution(1e-5, 1e-1, log=True),
            "n_layers": IntDistribution(1, 10),
            "dropout": FloatDistribution(0.0, 0.5),
            "activation": CategoricalDistribution(["relu", "gelu", "tanh"]),
        }

    def test_encoder_n_dims(self, mixed_space):
        encoder = SearchSpaceEncoder(mixed_space)
        assert encoder.n_dims == 4

    def test_encode_decode_roundtrip(self, mixed_space):
        encoder = SearchSpaceEncoder(mixed_space)
        params = {"lr": 1e-3, "n_layers": 5, "dropout": 0.2, "activation": "gelu"}

        encoded = encoder.encode_params(params)
        assert encoded.shape == (4,)
        assert np.all(encoded >= 0.0)
        assert np.all(encoded <= 1.0)

        decoded = encoder.decode_params(encoded)
        assert abs(decoded["dropout"] - 0.2) < 0.01
        assert decoded["n_layers"] == 5
        assert decoded["activation"] == "gelu"

    def test_encode_missing_params(self, mixed_space):
        encoder = SearchSpaceEncoder(mixed_space)
        encoded = encoder.encode_params({"lr": 1e-3})
        assert encoded.shape == (4,)
        # Missing params should default to 0.5
        assert encoded[1] == 0.5  # n_layers
        assert encoded[2] == 0.5  # dropout

    def test_encode_bounds(self, mixed_space):
        encoder = SearchSpaceEncoder(mixed_space)
        lower = {"lr": 1e-4, "dropout": 0.1}
        upper = {"lr": 1e-2, "dropout": 0.3}
        lo, hi = encoder.encode_bounds(lower, upper)
        assert lo.shape == (4,)
        assert hi.shape == (4,)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestApplyRegionBounds:
    """Test region bound application to search spaces."""

    def test_constrain_float_space(self):
        space = {"x": FloatDistribution(0.0, 1.0)}
        encoder = SearchSpaceEncoder(space)

        lower = np.array([0.2])
        upper = np.array([0.8])
        constrained = apply_region_bounds(space, encoder, lower, upper)

        assert constrained["x"].low >= 0.2 - 0.01
        assert constrained["x"].high <= 0.8 + 0.01

    def test_constrain_int_space(self):
        space = {"n": IntDistribution(1, 100)}
        encoder = SearchSpaceEncoder(space)

        lower = np.array([0.3])
        upper = np.array([0.7])
        constrained = apply_region_bounds(space, encoder, lower, upper)

        assert constrained["n"].low >= 1
        assert constrained["n"].high <= 100
        assert constrained["n"].low < constrained["n"].high

    def test_categorical_preserved(self):
        space = {"act": CategoricalDistribution(["relu", "gelu"])}
        encoder = SearchSpaceEncoder(space)

        lower = np.array([0.0])
        upper = np.array([1.0])
        constrained = apply_region_bounds(space, encoder, lower, upper)

        assert isinstance(constrained["act"], CategoricalDistribution)
        assert constrained["act"].choices == ("relu", "gelu")

    def test_narrow_bounds_expanded(self):
        space = {"x": FloatDistribution(0.0, 1.0)}
        encoder = SearchSpaceEncoder(space)

        # Very narrow bounds
        lower = np.array([0.5])
        upper = np.array([0.500001])
        constrained = apply_region_bounds(space, encoder, lower, upper)

        # Should be expanded to minimum width
        assert constrained["x"].high - constrained["x"].low > 0.01
