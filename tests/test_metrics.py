"""Tests for evaluation metrics."""

import numpy as np
import pytest

from trip_tpe.utils.metrics import (
    area_under_regret_curve,
    average_rank,
    convergence_curve,
    normalized_regret,
    normalized_regret_at_budget,
    wilcoxon_signed_rank_test,
)


class TestNormalizedRegret:
    """Test normalized regret computation."""

    def test_perfect_optimizer(self):
        """Optimizer that finds minimum on first try should have regret 0."""
        values = np.array([0.0, 0.5, 0.3, 0.8])
        regret = normalized_regret(values, y_min=0.0, y_max=1.0)
        assert regret[0] == 0.0
        assert np.all(regret == 0.0)  # Can't improve on 0.0

    def test_worst_optimizer(self):
        """Optimizer that never improves from worst should have regret 1."""
        values = np.array([1.0, 1.0, 1.0])
        regret = normalized_regret(values, y_min=0.0, y_max=1.0)
        np.testing.assert_array_almost_equal(regret, [1.0, 1.0, 1.0])

    def test_monotonically_decreasing(self):
        """Regret should be monotonically non-increasing."""
        values = np.random.RandomState(42).uniform(0, 1, size=50)
        regret = normalized_regret(values, y_min=0.0, y_max=1.0)
        diffs = np.diff(regret)
        assert np.all(diffs <= 1e-10)

    def test_regret_at_budget(self):
        values = np.array([1.0, 0.5, 0.3, 0.1, 0.05])
        r = normalized_regret_at_budget(values, y_min=0.0, y_max=1.0, budget=3)
        assert r == pytest.approx(0.3, abs=1e-6)

    def test_maximization(self):
        values = np.array([0.0, 0.5, 0.8, 0.6])
        regret = normalized_regret(values, y_min=0.0, y_max=1.0, minimize=False)
        assert regret[0] == 1.0  # Started at worst
        assert regret[-1] == pytest.approx(0.2, abs=1e-6)  # Best so far is 0.8


class TestAverageRank:
    """Test average rank computation."""

    def test_clear_winner(self):
        results = {
            "method_a": [0.1, 0.2, 0.15],
            "method_b": [0.5, 0.6, 0.55],
        }
        ranks = average_rank(results)
        assert ranks["method_a"] < ranks["method_b"]

    def test_tied_methods(self):
        results = {
            "a": [0.5, 0.5],
            "b": [0.5, 0.5],
        }
        ranks = average_rank(results)
        # Both should have same rank
        assert abs(ranks["a"] - ranks["b"]) < 0.01


class TestConvergenceCurve:
    """Test convergence curve computation."""

    def test_output_shape(self):
        trajectories = [np.random.rand(50) for _ in range(10)]
        mean, lower, upper = convergence_curve(trajectories, 0.0, 1.0, 50)
        assert mean.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)

    def test_ci_contains_mean(self):
        trajectories = [np.random.rand(50) for _ in range(10)]
        mean, lower, upper = convergence_curve(trajectories, 0.0, 1.0, 50)
        assert np.all(lower <= mean + 1e-8)
        assert np.all(upper >= mean - 1e-8)


class TestWilcoxon:
    """Test Wilcoxon signed-rank test."""

    def test_identical_distributions(self):
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        stat, pval = wilcoxon_signed_rank_test(a, b)
        assert pval == 1.0  # Identical → no difference

    def test_clearly_better(self):
        a = [0.1, 0.15, 0.12, 0.08, 0.11, 0.09, 0.13]
        b = [0.5, 0.6, 0.55, 0.7, 0.65, 0.58, 0.62]
        stat, pval = wilcoxon_signed_rank_test(a, b, alternative="less")
        assert pval < 0.05  # a is clearly better (lower)


class TestAURC:
    """Test Area Under Regret Curve."""

    def test_perfect_aurc(self):
        values = np.array([0.0] * 50)  # Perfect optimizer
        aurc = area_under_regret_curve(values, 0.0, 1.0, 50)
        assert aurc == pytest.approx(0.0, abs=1e-6)

    def test_aurc_bounded(self):
        values = np.random.RandomState(42).uniform(0, 1, size=100)
        aurc = area_under_regret_curve(values, 0.0, 1.0, 100)
        assert 0.0 <= aurc <= 1.0
