"""
Evaluation metrics for hyperparameter optimization benchmarking.

Primary metric: Normalized Regret across fixed evaluation budgets.
Supporting: convergence curves, average rank, critical difference diagrams.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def normalized_regret(
    observed_values: np.ndarray,
    y_min: float,
    y_max: float,
    minimize: bool = True,
) -> np.ndarray:
    """Compute normalized regret at each step of an optimization trajectory.

    Normalized regret maps the best-so-far value to [0, 1] where:
        - 0 = achieved the global optimum (y_min for minimization)
        - 1 = worst possible value (y_max for minimization)

    Args:
        observed_values: Array of objective values, shape (n_trials,).
        y_min: Known or estimated global minimum.
        y_max: Known or estimated global maximum (worst case).
        minimize: If True, lower is better.

    Returns:
        Array of normalized regret values, shape (n_trials,).
    """
    if minimize:
        best_so_far = np.minimum.accumulate(observed_values)
        regret = (best_so_far - y_min) / max(y_max - y_min, 1e-12)
    else:
        best_so_far = np.maximum.accumulate(observed_values)
        regret = (y_max - best_so_far) / max(y_max - y_min, 1e-12)

    return np.clip(regret, 0.0, 1.0)


def normalized_regret_at_budget(
    observed_values: np.ndarray,
    y_min: float,
    y_max: float,
    budget: int,
    minimize: bool = True,
) -> float:
    """Compute normalized regret at a specific evaluation budget.

    Args:
        observed_values: Array of objective values.
        y_min: Global minimum.
        y_max: Global maximum.
        budget: Number of evaluations at which to measure regret.
        minimize: If True, lower is better.

    Returns:
        Scalar normalized regret at the given budget.
    """
    regret_curve = normalized_regret(observed_values, y_min, y_max, minimize)
    idx = min(budget - 1, len(regret_curve) - 1)
    return float(regret_curve[idx])


def average_rank(
    results: Dict[str, List[float]],
) -> Dict[str, float]:
    """Compute average rank across multiple benchmark instances.

    Args:
        results: Mapping from method name to list of metric values
                 (one per benchmark instance). Lower is better.

    Returns:
        Mapping from method name to average rank (lower is better).
    """
    methods = list(results.keys())
    n_instances = len(next(iter(results.values())))

    all_ranks = {m: [] for m in methods}
    for i in range(n_instances):
        values = [(results[m][i], m) for m in methods]
        values.sort(key=lambda x: x[0])
        for rank, (_, method) in enumerate(values, 1):
            all_ranks[method].append(rank)

    return {m: float(np.mean(ranks)) for m, ranks in all_ranks.items()}


def wilcoxon_signed_rank_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alternative: str = "less",
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test between two methods.

    Tests whether method A is significantly better than method B.

    Args:
        values_a: Metric values for method A (one per instance).
        values_b: Metric values for method B (one per instance).
        alternative: "less" (A < B), "greater" (A > B), "two-sided".

    Returns:
        Tuple of (test_statistic, p_value).
    """
    a = np.array(values_a)
    b = np.array(values_b)
    diffs = a - b

    # Remove zero differences
    nonzero = np.abs(diffs) > 1e-12
    if nonzero.sum() < 3:
        return 0.0, 1.0  # Not enough data

    result = stats.wilcoxon(diffs[nonzero], alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def convergence_curve(
    trajectories: List[np.ndarray],
    y_min: float,
    y_max: float,
    max_budget: int,
    minimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean convergence curve with confidence intervals.

    Args:
        trajectories: List of objective value arrays from multiple seeds.
        y_min: Global minimum.
        y_max: Global maximum.
        max_budget: Maximum budget to plot.
        minimize: If True, lower is better.

    Returns:
        Tuple of (mean_regret, lower_ci, upper_ci), each shape (max_budget,).
    """
    regrets = []
    for traj in trajectories:
        reg = normalized_regret(traj[:max_budget], y_min, y_max, minimize)
        # Pad if trajectory is shorter than max_budget
        if len(reg) < max_budget:
            reg = np.pad(reg, (0, max_budget - len(reg)), constant_values=reg[-1])
        regrets.append(reg)

    regrets_arr = np.array(regrets)  # shape: (n_seeds, max_budget)
    mean = np.mean(regrets_arr, axis=0)
    std = np.std(regrets_arr, axis=0)
    n = len(trajectories)
    ci = 1.96 * std / np.sqrt(n)  # 95% confidence interval

    return mean, mean - ci, mean + ci


def area_under_regret_curve(
    observed_values: np.ndarray,
    y_min: float,
    y_max: float,
    budget: int,
    minimize: bool = True,
) -> float:
    """Compute the area under the normalized regret curve (AURC).

    AURC provides a single scalar summarizing optimization performance
    across all budgets up to the given limit.

    Args:
        observed_values: Array of objective values.
        y_min: Global minimum.
        y_max: Global maximum.
        budget: Maximum budget.
        minimize: If True, lower is better.

    Returns:
        Scalar AURC value (lower is better).
    """
    regret = normalized_regret(observed_values[:budget], y_min, y_max, minimize)
    return float(np.trapz(regret, dx=1.0) / budget)
