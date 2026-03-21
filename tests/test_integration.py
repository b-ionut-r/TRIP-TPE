"""End-to-end integration tests for TRIP-TPE.

Tests the full pipeline: data generation → model training → sampler usage.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TestEndToEndPipeline:
    """Integration tests verifying the complete TRIP-TPE workflow."""

    @pytest.fixture
    def trained_model_path(self, tmp_path):
        """Train a minimal model and return the checkpoint path."""
        from trip_tpe.models.region_proposal_transformer import RegionProposalTransformer
        from trip_tpe.data.trajectory_dataset import TrajectoryDataset
        from trip_tpe.data.preprocessing import TrajectoryPreprocessor
        from trip_tpe.training.loss import RegionProposalLoss

        # Generate minimal synthetic data
        preprocessor = TrajectoryPreprocessor(seed=42)
        rng = np.random.RandomState(42)
        pairs = []
        for i in range(20):
            n_trials = rng.randint(20, 40)
            configs = rng.rand(n_trials, 4).astype(np.float32)
            center = rng.rand(4).astype(np.float32)
            objectives = np.sum((configs - center)**2, axis=1).astype(np.float32)
            pairs.extend(preprocessor.process_trajectory(configs, objectives, f"int_{i}"))

        dataset = TrajectoryDataset(pairs, max_seq_len=50, hp_dim=8)

        # Create minimal model
        model = RegionProposalTransformer(
            hp_dim=8, embed_dim=64, num_heads=2, num_layers=1,
            ff_dim=128, max_seq_len=50, dropout=0.0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = RegionProposalLoss()

        # Train for a few steps
        model.train()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, collate_fn=TrajectoryDataset.collate_fn
        )
        for batch in loader:
            predictions = model(batch["input_seq"], batch["attention_mask"])
            losses = loss_fn(
                predictions, batch["target_lower"], batch["target_upper"], batch["dim_mask"]
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            break  # Just one batch

        # Save checkpoint
        ckpt_path = str(tmp_path / "test_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "hp_dim": 8, "embed_dim": 64, "num_heads": 2,
                "num_layers": 1, "ff_dim": 128, "max_seq_len": 50, "dropout": 0.0,
            },
        }, ckpt_path)

        return ckpt_path

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_full_pipeline_with_model(self, trained_model_path):
        """Test complete pipeline: trained model → sampler → optimization."""
        from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler

        sampler = TRIPTPESampler(
            model_path=trained_model_path,
            n_warmup_trials=3,
            requery_interval=5,
            seed=42,
            hp_dim=8,
            device="cpu",
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            z = trial.suggest_float("z", 0, 1)
            return (x - 0.3)**2 + (y - 0.7)**2 + (z - 0.5)**2

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=20)

        assert len(study.trials) == 20
        assert study.best_value is not None
        assert sampler.is_model_loaded

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_sampler_without_model_still_works(self):
        """Verify graceful degradation without a trained model."""
        from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler

        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return (x - 0.5)**2

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=30)

        assert len(study.trials) == 30
        assert study.best_value < 0.1

    def test_model_save_load_roundtrip(self, tmp_path):
        """Test model checkpoint save/load preserves architecture."""
        from trip_tpe.models.region_proposal_transformer import RegionProposalTransformer

        model = RegionProposalTransformer(
            hp_dim=8, embed_dim=64, num_heads=2, num_layers=1,
            ff_dim=128, max_seq_len=50,
        )

        # Save
        ckpt_path = str(tmp_path / "roundtrip.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "hp_dim": 8, "embed_dim": 64, "num_heads": 2,
                "num_layers": 1, "ff_dim": 128, "max_seq_len": 50,
            },
        }, ckpt_path)

        # Load
        checkpoint = torch.load(ckpt_path, weights_only=False)
        loaded_model = RegionProposalTransformer(**checkpoint["model_config"])
        loaded_model.load_state_dict(checkpoint["model_state_dict"])

        # Compare outputs
        test_input = torch.randn(1, 10, 9)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out1 = model(test_input)
            out2 = loaded_model(test_input)

        torch.testing.assert_close(out1["pred_lower"], out2["pred_lower"])
        torch.testing.assert_close(out1["pred_upper"], out2["pred_upper"])

    def test_config_to_model_to_training(self):
        """Test config → model instantiation → training step."""
        from trip_tpe.utils.config import TRIPTPEConfig
        from trip_tpe.models.region_proposal_transformer import RegionProposalTransformer
        from trip_tpe.training.loss import RegionProposalLoss

        config = TRIPTPEConfig()
        config.transformer.num_layers = 1
        config.transformer.embed_dim = 64
        config.transformer.num_heads = 2
        config.transformer.ff_dim = 128
        config.transformer.hp_input_dim = 8

        model = RegionProposalTransformer(
            hp_dim=config.transformer.hp_input_dim,
            embed_dim=config.transformer.embed_dim,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            ff_dim=config.transformer.ff_dim,
        )

        loss_fn = RegionProposalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

        # Fake batch
        batch_size = 4
        hp_dim = 8
        seq_len = 20
        input_seq = torch.randn(batch_size, seq_len, hp_dim + 1)
        mask = torch.ones(batch_size, seq_len)
        target_lower = torch.rand(batch_size, hp_dim) * 0.5
        target_upper = target_lower + torch.rand(batch_size, hp_dim) * 0.5
        dim_mask = torch.ones(batch_size, hp_dim)

        # Forward + backward
        model.train()
        predictions = model(input_seq, mask)
        losses = loss_fn(predictions, target_lower, target_upper, dim_mask)

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        assert losses["loss"].item() > 0
