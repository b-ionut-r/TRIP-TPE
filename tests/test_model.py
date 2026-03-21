"""Tests for the Region Proposal Transformer model."""

import pytest
import torch
import numpy as np

from trip_tpe.models.region_proposal_transformer import (
    RegionProposalTransformer,
    TrialEmbedding,
    LearnedPositionalEncoding,
    RegionProposalHead,
)


class TestTrialEmbedding:
    """Test the trial embedding module."""

    def test_output_shape(self):
        emb = TrialEmbedding(hp_dim=16, obj_dim=1, embed_dim=256)
        x = torch.randn(4, 20, 17)  # batch=4, seq=20, hp_dim+obj_dim=17
        out = emb(x, hp_dim=16)
        assert out.shape == (4, 20, 256)

    def test_different_dims(self):
        emb = TrialEmbedding(hp_dim=8, obj_dim=1, embed_dim=128)
        x = torch.randn(2, 10, 9)
        out = emb(x, hp_dim=8)
        assert out.shape == (2, 10, 128)


class TestRegionProposalTransformer:
    """Test the full Transformer model."""

    @pytest.fixture
    def model(self):
        return RegionProposalTransformer(
            hp_dim=8,
            obj_dim=1,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_seq_len=50,
            dropout=0.0,
        )

    def test_forward_shape(self, model):
        batch_size = 4
        seq_len = 20
        input_seq = torch.randn(batch_size, seq_len, 9)  # hp_dim + obj_dim
        mask = torch.ones(batch_size, seq_len)

        output = model(input_seq, attention_mask=mask)

        assert "alpha" in output
        assert "beta" in output
        assert "pred_lower" in output
        assert "pred_upper" in output
        assert "cls_embedding" in output

        assert output["alpha"].shape == (batch_size, 8)
        assert output["pred_lower"].shape == (batch_size, 8)
        assert output["cls_embedding"].shape == (batch_size, 128)

    def test_alpha_beta_positive(self, model):
        input_seq = torch.randn(2, 10, 9)
        output = model(input_seq)
        assert (output["alpha"] > 0).all()
        assert (output["beta"] > 0).all()

    def test_bounds_in_unit_interval(self, model):
        input_seq = torch.randn(2, 10, 9)
        output = model(input_seq)
        assert (output["pred_lower"] >= 0).all()
        assert (output["pred_lower"] <= 1).all()
        assert (output["pred_upper"] >= 0).all()
        assert (output["pred_upper"] <= 1).all()

    def test_lower_less_than_upper(self, model):
        input_seq = torch.randn(2, 10, 9)
        output = model(input_seq)
        assert (output["pred_lower"] <= output["pred_upper"]).all()

    def test_predict_region(self, model):
        input_seq = torch.randn(2, 10, 9)
        mask = torch.ones(2, 10)
        lower, upper = model.predict_region(input_seq, mask, confidence_level=0.9)
        assert lower.shape == (2, 8)
        assert upper.shape == (2, 8)
        assert (lower >= 0).all()
        assert (upper <= 1).all()
        assert (lower <= upper).all()

    def test_masked_input(self, model):
        batch_size = 2
        seq_len = 20
        input_seq = torch.randn(batch_size, seq_len, 9)
        mask = torch.zeros(batch_size, seq_len)
        mask[:, :10] = 1.0  # Only first 10 tokens are valid

        output = model(input_seq, attention_mask=mask)
        assert output["pred_lower"].shape == (batch_size, 8)

    def test_parameter_count(self, model):
        n_params = model.count_parameters()
        assert n_params > 0
        # Lightweight model should be well under 10M
        assert n_params < 5_000_000

    def test_memory_estimate(self, model):
        mem = model.memory_estimate_mb(batch_size=32, seq_len=50)
        assert "estimated_total_mb" in mem
        assert mem["estimated_total_mb"] > 0

    def test_default_model_parameter_count(self):
        """Verify the default architecture is ~10M parameters."""
        model = RegionProposalTransformer(
            hp_dim=16, embed_dim=256, num_heads=8,
            num_layers=4, ff_dim=1024,
        )
        n_params = model.count_parameters()
        # Should be in the ~9-11M range
        assert 5_000_000 < n_params < 15_000_000, f"Got {n_params:,} params"

    def test_gradient_flow(self, model):
        """Ensure gradients flow through the entire model."""
        input_seq = torch.randn(2, 10, 9, requires_grad=False)
        mask = torch.ones(2, 10)

        output = model(input_seq, mask)
        loss = output["pred_lower"].sum() + output["pred_upper"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
