"""Tests for the region proposal loss function."""

import pytest
import torch

from trip_tpe.training.loss import RegionProposalLoss


class TestRegionProposalLoss:
    """Test the multi-objective loss function."""

    @pytest.fixture
    def loss_fn(self):
        return RegionProposalLoss()

    @pytest.fixture
    def sample_predictions(self):
        batch_size = 4
        hp_dim = 8
        return {
            "alpha": torch.ones(batch_size, hp_dim) * 2.0,
            "beta": torch.ones(batch_size, hp_dim) * 2.0,
            "pred_lower": torch.full((batch_size, hp_dim), 0.2),
            "pred_upper": torch.full((batch_size, hp_dim), 0.8),
        }

    @pytest.fixture
    def sample_targets(self):
        batch_size = 4
        hp_dim = 8
        return {
            "target_lower": torch.full((batch_size, hp_dim), 0.25),
            "target_upper": torch.full((batch_size, hp_dim), 0.75),
            "dim_mask": torch.ones(batch_size, hp_dim),
        }

    def test_loss_computes(self, loss_fn, sample_predictions, sample_targets):
        losses = loss_fn(
            sample_predictions,
            sample_targets["target_lower"],
            sample_targets["target_upper"],
            sample_targets["dim_mask"],
        )
        assert "loss" in losses
        assert losses["loss"].item() >= 0.0
        assert not torch.isnan(losses["loss"])

    def test_loss_components_present(self, loss_fn, sample_predictions, sample_targets):
        losses = loss_fn(
            sample_predictions,
            sample_targets["target_lower"],
            sample_targets["target_upper"],
            sample_targets["dim_mask"],
        )
        for key in ["bound_loss", "coverage_loss", "tightness_loss", "kl_loss"]:
            assert key in losses
            assert not torch.isnan(losses[key])

    def test_perfect_prediction_low_loss(self, loss_fn):
        batch_size = 2
        hp_dim = 4
        target_lower = torch.full((batch_size, hp_dim), 0.3)
        target_upper = torch.full((batch_size, hp_dim), 0.7)
        dim_mask = torch.ones(batch_size, hp_dim)

        # Perfect predictions: exactly match targets
        predictions = {
            "alpha": torch.ones(batch_size, hp_dim) * 2.0,
            "beta": torch.ones(batch_size, hp_dim) * 2.0,
            "pred_lower": target_lower.clone(),
            "pred_upper": target_upper.clone(),
        }

        losses = loss_fn(predictions, target_lower, target_upper, dim_mask)

        # Bound loss should be near 0, coverage loss should be 0
        assert losses["bound_loss"].item() < 0.01
        assert losses["coverage_loss"].item() < 0.01

    def test_dim_mask_respected(self, loss_fn):
        batch_size = 2
        hp_dim = 8
        target_lower = torch.full((batch_size, hp_dim), 0.3)
        target_upper = torch.full((batch_size, hp_dim), 0.7)

        predictions = {
            "alpha": torch.ones(batch_size, hp_dim) * 2.0,
            "beta": torch.ones(batch_size, hp_dim) * 2.0,
            "pred_lower": torch.full((batch_size, hp_dim), 0.0),
            "pred_upper": torch.full((batch_size, hp_dim), 1.0),
        }

        # All dims active
        dim_mask_full = torch.ones(batch_size, hp_dim)
        loss_full = loss_fn(predictions, target_lower, target_upper, dim_mask_full)

        # Only first 4 dims active
        dim_mask_half = torch.zeros(batch_size, hp_dim)
        dim_mask_half[:, :4] = 1.0
        loss_half = loss_fn(predictions, target_lower, target_upper, dim_mask_half)

        # Losses should differ when dim mask changes
        # (tightness loss should be lower with fewer active dims)
        assert loss_full["tightness_loss"].item() != loss_half["tightness_loss"].item()

    def test_gradient_flows(self, loss_fn):
        batch_size = 2
        hp_dim = 4
        predictions = {
            "alpha": torch.ones(batch_size, hp_dim, requires_grad=True) * 2.0,
            "beta": torch.ones(batch_size, hp_dim, requires_grad=True) * 2.0,
            "pred_lower": torch.full((batch_size, hp_dim), 0.2, requires_grad=True),
            "pred_upper": torch.full((batch_size, hp_dim), 0.8, requires_grad=True),
        }
        target_lower = torch.full((batch_size, hp_dim), 0.3)
        target_upper = torch.full((batch_size, hp_dim), 0.7)
        dim_mask = torch.ones(batch_size, hp_dim)

        losses = loss_fn(predictions, target_lower, target_upper, dim_mask)
        losses["loss"].backward()

        assert predictions["pred_lower"].grad is not None
        assert predictions["pred_upper"].grad is not None
