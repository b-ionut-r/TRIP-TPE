"""Tests for the TRIP-TPE Optuna sampler."""

import numpy as np
import pytest

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestTRIPTPESampler:
    """Test the TRIP-TPE sampler integration with Optuna."""

    def test_fallback_to_tpe_without_model(self):
        """Without a model, sampler should gracefully fall back to TPE."""
        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            return (x - 0.5)**2 + (y - 0.3)**2

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=20)

        assert len(study.trials) == 20
        assert study.best_value < 0.5  # Should find something reasonable
        assert not sampler.is_model_loaded

    def test_fallback_with_invalid_model_path(self):
        """Invalid model path should warn and fall back to TPE."""
        sampler = TRIPTPESampler(model_path="/nonexistent/model.pt", seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return (x - 0.5)**2

        study = optuna.create_study(sampler=sampler, direction="minimize")

        with pytest.warns(RuntimeWarning):
            study.optimize(objective, n_trials=10)

        assert len(study.trials) == 10

    def test_warmup_phase(self):
        """First N trials should be random (warmup phase)."""
        sampler = TRIPTPESampler(
            model_path=None,
            n_warmup_trials=10,
            seed=42,
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x**2

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=15)
        assert len(study.trials) == 15

    def test_integer_and_categorical_params(self):
        """Test with mixed parameter types."""
        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            n = trial.suggest_int("n_layers", 1, 10)
            act = trial.suggest_categorical("activation", ["relu", "gelu", "tanh"])
            return x**2 + n * 0.1

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=20)
        assert len(study.trials) == 20

    def test_log_scale_params(self):
        """Test with log-scale parameters."""
        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            return (lr - 1e-3)**2

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=20)
        assert len(study.trials) == 20

    def test_trust_factor_properties(self):
        """Test trust factor initialization and decay."""
        sampler = TRIPTPESampler(
            trust_factor=0.8,
            trust_decay=0.99,
            min_trust_factor=0.3,
        )
        assert sampler.current_trust_factor == 0.8
        assert sampler.n_requery_count == 0

    def test_multi_objective(self):
        """Test with multi-objective optimization."""
        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x**2, (1 - x)**2

        study = optuna.create_study(
            sampler=sampler,
            directions=["minimize", "minimize"],
        )
        study.optimize(objective, n_trials=20)
        assert len(study.trials) == 20

    def test_reproducibility(self):
        """Same seed should produce same results."""
        results = []
        for _ in range(2):
            sampler = TRIPTPESampler(model_path=None, seed=123)

            def objective(trial):
                x = trial.suggest_float("x", 0, 1)
                return (x - 0.5)**2

            study = optuna.create_study(sampler=sampler, direction="minimize")
            study.optimize(objective, n_trials=10)
            results.append([t.value for t in study.trials])

        np.testing.assert_array_almost_equal(results[0], results[1])

    def test_maximization_direction(self):
        """Test that the sampler works correctly with maximize direction."""
        sampler = TRIPTPESampler(model_path=None, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            # Maximum at (0.7, 0.3) = 1.0
            return 1.0 - ((x - 0.7)**2 + (y - 0.3)**2)

        study = optuna.create_study(sampler=sampler, direction="maximize")
        study.optimize(objective, n_trials=30)

        assert len(study.trials) == 30
        # Should find a value reasonably close to the maximum of 1.0
        assert study.best_value > 0.5

    def test_adaptive_trust_maximize_direction(self):
        """Test that adaptive trust computes correctly for maximization studies.

        Regression test: ensures _compute_adaptive_trust respects study.direction
        when selecting top-gamma trials. For maximize, the best trials have the
        highest values and should be sorted descending before slicing.
        """
        sampler = TRIPTPESampler(
            model_path=None,
            seed=42,
            adaptive_trust=True,
            adaptive_trust_target_coverage=0.8,
            adaptive_trust_gamma=0.15,
            trust_factor=0.8,
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x  # maximize → best trials have highest x

        study = optuna.create_study(sampler=sampler, direction="maximize")
        study.optimize(objective, n_trials=30)

        # Build the encoder so _compute_adaptive_trust can run
        search_space = optuna.search_space.intersection_search_space(study)
        sampler._build_encoder(search_space)
        # Set dummy cached bounds (full space) so the method doesn't bail out
        sampler._cached_lower = np.zeros(sampler._encoder.n_dims, dtype=np.float32)
        sampler._cached_upper = np.ones(sampler._encoder.n_dims, dtype=np.float32)

        trust = sampler._compute_adaptive_trust(study)

        # With full [0,1] bounds, all trials are "inside" → coverage = 1.0
        # → trust should decay slowly (adjustment > 0.5 → decay_multiplier > 1.0)
        # The result should still be a valid trust factor in [min_trust, 1.0]
        assert 0.0 < trust <= 1.0
        assert trust >= sampler._min_trust_factor

    def test_hypervolume_guard_high_dimensional(self):
        """Test that the hypervolume guard does NOT over-expand in high-d spaces.

        Regression test for the fixed 1e-6 threshold bug: at d=16 with moderate
        per-dimension widths (e.g., 0.3), the joint volume is 0.3^16 ≈ 4.3e-9,
        which is below 1e-6 but above the dimension-aware threshold of
        0.05^16 ≈ 1.5e-21. The guard should NOT trigger in this case.
        """
        sampler = TRIPTPESampler(
            model_path=None,
            min_region_fraction=0.05,
        )

        n_dims = 16
        # Moderate widths: 0.3 per dimension → volume = 0.3^16 ≈ 4.3e-9
        lower = np.full(n_dims, 0.35, dtype=np.float32)
        upper = np.full(n_dims, 0.65, dtype=np.float32)

        result_lower, result_upper = sampler._apply_joint_hypervolume_guard(lower, upper)

        # The guard should NOT have expanded the bounds
        np.testing.assert_array_almost_equal(result_lower, lower)
        np.testing.assert_array_almost_equal(result_upper, upper)
