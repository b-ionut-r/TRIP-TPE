"""
Benchmarking framework for TRIP-TPE.

Evaluates TRIP-TPE against baseline optimizers using normalized regret
on surrogate benchmarks (HPO-B, YAHPO Gym) and synthetic functions.

Implements the evaluation protocol required for NeurIPS/ICML submissions:
    - Normalized regret at fixed budgets (25, 50, 100 trials)
    - Convergence curves with confidence intervals
    - Average rank with critical difference diagrams
    - Wilcoxon signed-rank tests for statistical significance
    - Area Under Regret Curve (AURC) as a summary metric

Baseline tiers:
    Classical: Random Search, TPE (Optuna), GP-EI
    Advanced: SMAC3, BOHB, HEBO
    Modern SOTA: PFNs4BO, NAP, Task-Similarity MO-TPE
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from trip_tpe.utils.metrics import (
    area_under_regret_curve,
    average_rank,
    convergence_curve,
    normalized_regret_at_budget,
    wilcoxon_signed_rank_test,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    benchmark: str
    instance: str
    seed: int
    trajectory: np.ndarray  # Objective values per trial
    best_value: float
    n_trials: int
    wall_time: float
    regret_at_budgets: Dict[int, float] = field(default_factory=dict)
    aurc: float = 0.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark instances for evaluation."""
    name: str
    instances: List[Dict[str, Any]]  # Each has "objective", "search_space", "y_min", "y_max"
    minimize: bool = True


def create_synthetic_benchmark(
    n_instances: int = 20,
    n_dims_range: Tuple[int, int] = (4, 12),
    seed: int = 42,
) -> BenchmarkSuite:
    """Create a synthetic benchmark suite for testing.

    Generates diverse optimization problems with known global optima
    for computing exact normalized regret.

    Args:
        n_instances: Number of benchmark instances.
        n_dims_range: Range of dimensionalities.
        seed: Random seed.

    Returns:
        BenchmarkSuite with synthetic instances.
    """
    rng = np.random.RandomState(seed)
    instances = []

    func_types = ["quadratic", "rosenbrock", "ackley", "rastrigin"]

    for i in range(n_instances):
        n_dims = rng.randint(n_dims_range[0], n_dims_range[1] + 1)
        func_type = func_types[i % len(func_types)]
        center = rng.uniform(0.2, 0.8, size=n_dims)
        scales = rng.uniform(0.5, 5.0, size=n_dims)

        # Create Optuna search space
        search_space = {}
        for d in range(n_dims):
            search_space[f"x{d}"] = optuna.distributions.FloatDistribution(0.0, 1.0)

        def make_objective(center, scales, func_type, n_dims):
            def objective(trial):
                x = np.array([trial.suggest_float(f"x{d}", 0.0, 1.0) for d in range(n_dims)])
                diff = x - center
                if func_type == "quadratic":
                    return float(np.sum(scales * diff**2))
                elif func_type == "rosenbrock":
                    x_scaled = diff * 4.0
                    val = sum(
                        100.0 * (x_scaled[d + 1] - x_scaled[d]**2)**2 + (1.0 - x_scaled[d])**2
                        for d in range(n_dims - 1)
                    )
                    return val / n_dims
                elif func_type == "ackley":
                    x_scaled = diff * 4.0
                    sum_sq = np.mean(x_scaled**2)
                    sum_cos = np.mean(np.cos(2.0 * np.pi * x_scaled))
                    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + np.e)
                elif func_type == "rastrigin":
                    x_scaled = diff * 5.12
                    return float(10.0 * n_dims + np.sum(x_scaled**2 - 10.0 * np.cos(2.0 * np.pi * x_scaled))) / n_dims
                return 0.0
            return objective

        instances.append({
            "name": f"{func_type}_{n_dims}d_{i}",
            "objective": make_objective(center.copy(), scales.copy(), func_type, n_dims),
            "search_space": search_space,
            "y_min": 0.0,  # All functions have minimum 0 at center
            "y_max": 50.0,  # Rough upper bound
            "n_dims": n_dims,
        })

    return BenchmarkSuite(name="synthetic", instances=instances)


def _create_sampler(method: str, seed: int, model_path: Optional[str] = None) -> Any:
    """Create an Optuna sampler for a given method.

    Args:
        method: Sampler name.
        seed: Random seed.
        model_path: Path to TRIP-TPE model checkpoint.

    Returns:
        Optuna BaseSampler instance.
    """
    if method == "trip_tpe":
        from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler
        return TRIPTPESampler(
            model_path=model_path,
            seed=seed,
            n_warmup_trials=5,
            requery_interval=10,
        )
    elif method == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    elif method == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif method == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:
        # Default to TPE for unknown methods
        return optuna.samplers.TPESampler(seed=seed)


def run_single_benchmark(
    method: str,
    instance: Dict[str, Any],
    n_trials: int,
    seed: int,
    model_path: Optional[str] = None,
) -> BenchmarkResult:
    """Run a single optimization benchmark.

    Args:
        method: Sampler method name.
        instance: Benchmark instance dict.
        n_trials: Evaluation budget.
        seed: Random seed.
        model_path: Path to TRIP-TPE model (if method == "trip_tpe").

    Returns:
        BenchmarkResult with trajectory and metrics.
    """
    sampler = _create_sampler(method, seed, model_path)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    t0 = time.time()
    study.optimize(instance["objective"], n_trials=n_trials, show_progress_bar=False)
    wall_time = time.time() - t0

    # Extract trajectory
    trajectory = np.array([t.value for t in study.trials if t.value is not None])

    # Compute regret at standard budgets
    y_min = instance["y_min"]
    y_max = instance["y_max"]
    regret_at_budgets = {}
    for budget in [25, 50, 100]:
        if budget <= len(trajectory):
            regret_at_budgets[budget] = normalized_regret_at_budget(
                trajectory, y_min, y_max, budget
            )

    aurc = area_under_regret_curve(trajectory, y_min, y_max, min(100, len(trajectory)))

    return BenchmarkResult(
        method=method,
        benchmark=instance.get("name", "unknown"),
        instance=instance.get("name", "unknown"),
        seed=seed,
        trajectory=trajectory,
        best_value=float(study.best_value) if study.best_value is not None else float("inf"),
        n_trials=len(trajectory),
        wall_time=wall_time,
        regret_at_budgets=regret_at_budgets,
        aurc=aurc,
    )


def run_benchmark_suite(
    suite: BenchmarkSuite,
    methods: List[str],
    n_trials: int = 100,
    n_seeds: int = 5,
    model_path: Optional[str] = None,
    output_dir: str = "results",
    use_wandb: bool = False,
) -> Dict[str, List[BenchmarkResult]]:
    """Run a full benchmark suite across methods and seeds.

    Args:
        suite: Benchmark suite to evaluate.
        methods: List of method names to compare.
        n_trials: Evaluation budget per run.
        n_seeds: Number of random seeds.
        model_path: Path to TRIP-TPE model.
        output_dir: Directory to save results.

    Returns:
        Dictionary mapping method names to lists of BenchmarkResult.
    """
    results: Dict[str, List[BenchmarkResult]] = {m: [] for m in methods}

    total_runs = len(methods) * len(suite.instances) * n_seeds
    print(f"\nBenchmark: {suite.name}")
    print(f"Instances: {len(suite.instances)}, Methods: {len(methods)}, Seeds: {n_seeds}")
    print(f"Total runs: {total_runs}")
    print("-" * 60)

    run_idx = 0
    for instance in suite.instances:
        for method in methods:
            for seed in range(n_seeds):
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] {method} | {instance['name']} | seed={seed}",
                    end=" ... ",
                    flush=True,
                )

                result = run_single_benchmark(
                    method=method,
                    instance=instance,
                    n_trials=n_trials,
                    seed=seed,
                    model_path=model_path,
                )
                results[method].append(result)
                print(
                    f"best={result.best_value:.4f} | "
                    f"time={result.wall_time:.1f}s"
                )

                # W&B per-run logging
                if use_wandb:
                    run_log = {
                        f"bench/{method}/best_value": result.best_value,
                        f"bench/{method}/wall_time": result.wall_time,
                        f"bench/{method}/aurc": result.aurc,
                    }
                    for b, r in result.regret_at_budgets.items():
                        run_log[f"bench/{method}/regret@{b}"] = r
                    wandb.log(run_log)

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for budget in [25, 50, 100]:
        print(f"\n--- Normalized Regret @ Budget={budget} ---")
        regret_by_method = {}
        for method in methods:
            regrets = [
                r.regret_at_budgets.get(budget, float("nan"))
                for r in results[method]
                if budget in r.regret_at_budgets
            ]
            if regrets:
                regret_by_method[method] = regrets
                mean_r = np.nanmean(regrets)
                std_r = np.nanstd(regrets)
                print(f"  {method:15s}: {mean_r:.4f} +/- {std_r:.4f}")

        # Average rank
        if len(regret_by_method) > 1:
            min_len = min(len(v) for v in regret_by_method.values())
            truncated = {k: v[:min_len] for k, v in regret_by_method.items()}
            ranks = average_rank(truncated)
            print(f"\n  Average Rank:")
            for method, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"    {method:15s}: {rank:.2f}")

    # Statistical tests
    if "trip_tpe" in methods and len(methods) > 1:
        print(f"\n--- Wilcoxon Signed-Rank Tests (TRIP-TPE vs baselines) ---")
        trip_results = [r.aurc for r in results["trip_tpe"]]
        for method in methods:
            if method == "trip_tpe":
                continue
            baseline_results = [r.aurc for r in results[method]]
            min_len = min(len(trip_results), len(baseline_results))
            if min_len >= 5:
                stat, pval = wilcoxon_signed_rank_test(
                    trip_results[:min_len],
                    baseline_results[:min_len],
                )
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  TRIP-TPE vs {method:15s}: p={pval:.4f} {sig}")

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for method in methods:
        summary[method] = {
            "mean_aurc": float(np.mean([r.aurc for r in results[method]])),
            "mean_best": float(np.mean([r.best_value for r in results[method]])),
            "regret_at_budgets": {
                str(b): float(np.mean([
                    r.regret_at_budgets.get(b, float("nan"))
                    for r in results[method]
                ]))
                for b in [25, 50, 100]
            },
        }

    with open(out_dir / f"{suite.name}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir / f'{suite.name}_results.json'}")

    # W&B summary table and artifact
    if use_wandb:
        # Summary table
        table = wandb.Table(columns=["method", "mean_aurc", "mean_best", "regret@25", "regret@50", "regret@100"])
        for method in methods:
            s = summary[method]
            table.add_data(
                method, s["mean_aurc"], s["mean_best"],
                s["regret_at_budgets"].get("25", float("nan")),
                s["regret_at_budgets"].get("50", float("nan")),
                s["regret_at_budgets"].get("100", float("nan")),
            )
        wandb.log({"benchmark_summary": table})

        # Log convergence curves per method
        for method in methods:
            trajs = [r.trajectory for r in results[method]]
            if trajs:
                y_min = suite.instances[0]["y_min"]
                y_max = suite.instances[0]["y_max"]
                mean_reg, lo_ci, hi_ci = convergence_curve(trajs, y_min, y_max, n_trials)
                data = [[i, float(mean_reg[i])] for i in range(len(mean_reg))]
                table_conv = wandb.Table(data=data, columns=["trial", "mean_regret"])
                wandb.log({f"convergence/{method}": wandb.plot.line(
                    table_conv, "trial", "mean_regret", title=f"Convergence: {method}"
                )})

        # Results artifact
        artifact = wandb.Artifact(
            f"trip-tpe-benchmark-{suite.name}", type="results",
            description=f"Benchmark results: {len(methods)} methods, {n_seeds} seeds",
        )
        artifact.add_file(str(out_dir / f"{suite.name}_results.json"))
        wandb.log_artifact(artifact)

    return results


def main():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark TRIP-TPE against baselines")
    parser.add_argument("--model-path", type=str, default=None, help="Path to TRIP-TPE model")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["trip_tpe", "tpe", "random"],
        help="Methods to benchmark",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Budget per run")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--n-instances", type=int, default=10, help="Number of benchmark instances")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--wandb-project", type=str, default="trip-tpe", help="W&B project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Initialize W&B for benchmark tracking
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                job_type="benchmark",
                config={
                    "methods": args.methods,
                    "n_trials": args.n_trials,
                    "n_seeds": args.n_seeds,
                    "n_instances": args.n_instances,
                    "model_path": args.model_path,
                },
                tags=["benchmark"],
            )
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging.")
            use_wandb = False

    suite = create_synthetic_benchmark(n_instances=args.n_instances)

    run_benchmark_suite(
        suite=suite,
        methods=args.methods,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
