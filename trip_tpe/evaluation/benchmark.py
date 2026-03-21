"""
Benchmarking framework for TRIP-TPE.

Evaluates TRIP-TPE against baseline optimizers using normalized regret
on real surrogate benchmarks and synthetic functions.

Three benchmark tiers, each testing a different capability:

    1. YAHPO Gym (continuous, real): Fitted ML surrogate models that accept
       arbitrary configurations. Smooth, high-fidelity. Tests whether the
       region proposal Transformer genuinely improves continuous HPO.
       This is the primary tier for paper results.

    2. HPO-B (tabular, real): Pre-evaluated lookup tables of (config, value)
       pairs from real ML experiments. The optimizer proposes configs in
       continuous space, which are discretized to the nearest table entry.
       Each entry can only be "evaluated" once (no repeats). Regret is
       computed against the table's true best — no fictional y_min.
       This tier validates generalization to discrete selection tasks.

    3. Synthetic (continuous, analytical): Known analytical functions with
       exact global optima. Used for sanity checks and ablations only.

Data leakage prevention:
    - HPO-B training uses meta_train split; benchmarking uses meta_test.
    - YAHPO training instances and benchmark instances are disjoint;
      the held-out set is recorded in the training data manifest.
    - The generate_trajectories module saves a manifest of all (source,
      scenario, instance_id) tuples used for training, which this module
      reads to enforce the split.

Evaluation protocol (NeurIPS/ICML standard):
    - Normalized regret at fixed budgets (25, 50, 100 trials)
    - Convergence curves with 95% confidence intervals
    - Average rank with critical difference diagrams
    - Wilcoxon signed-rank tests for statistical significance
    - Area Under Regret Curve (AURC) as a summary scalar

Baseline tiers:
    Classical: Random Search, TPE (Optuna), GP-EI
    Advanced: SMAC3, BOHB, HEBO
    Modern SOTA: PFNs4BO, NAP, Task-Similarity MO-TPE
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


# ============================================================================
# Data structures
# ============================================================================

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
class BenchmarkInstance:
    """A single benchmark problem instance."""
    name: str
    objective: Callable  # Optuna-compatible trial → float
    search_space: Dict[str, Any]  # Optuna distributions
    y_min: float  # Best achievable value (for regret normalization)
    y_max: float  # Worst-case reference (for regret normalization)
    n_dims: int
    minimize: bool = True
    source: str = "unknown"  # "synthetic", "hpob", "yahpo"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark instances for evaluation."""
    name: str
    instances: List[BenchmarkInstance]
    description: str = ""


# ============================================================================
# Training manifest (leakage prevention)
# ============================================================================

def load_training_manifest(manifest_path: str) -> Set[str]:
    """Load the set of instance IDs used during training.

    The data generation pipeline saves a JSON manifest listing every
    (source, scenario, instance_id) used for training. This function
    loads those IDs so the benchmark can exclude them.

    Args:
        manifest_path: Path to training_manifest.json.

    Returns:
        Set of "source/scenario/instance_id" strings.
    """
    path = Path(manifest_path)
    if not path.exists():
        return set()

    with open(path) as f:
        manifest = json.load(f)

    ids = set()
    for entry in manifest.get("instances",[]):
        key = f"{entry.get('source', '')}/{entry.get('scenario', '')}/{entry.get('instance_id', '')}"
        ids.add(key)
    return ids


# ============================================================================
# Tier 3: Synthetic benchmark (sanity check / ablation)
# ============================================================================

def create_synthetic_benchmark(
    n_instances: int = 20,
    n_dims_range: Tuple[int, int] = (4, 12),
    seed: int = 42,
) -> BenchmarkSuite:
    """Create a synthetic benchmark suite with known global optima.

    Args:
        n_instances: Number of benchmark instances.
        n_dims_range: Range of dimensionalities.
        seed: Random seed.

    Returns:
        BenchmarkSuite with synthetic instances.
    """
    rng = np.random.RandomState(seed)
    instances =[]

    func_types = ["quadratic", "rosenbrock", "ackley", "rastrigin"]

    for i in range(n_instances):
        n_dims = rng.randint(n_dims_range[0], n_dims_range[1] + 1)
        func_type = func_types[i % len(func_types)]
        center = rng.uniform(0.2, 0.8, size=n_dims)
        scales = rng.uniform(0.5, 5.0, size=n_dims)

        search_space = {
            f"x{d}": optuna.distributions.FloatDistribution(0.0, 1.0)
            for d in range(n_dims)
        }

        def make_objective(center, scales, func_type, n_dims):
            def objective(trial):
                x = np.array([trial.suggest_float(f"x{d}", 0.0, 1.0) for d in range(n_dims)])
                diff = x - center
                if func_type == "quadratic":
                    return float(np.sum(scales * diff**2))
                elif func_type == "rosenbrock":
                    xs = diff * 4.0
                    val = sum(100.0 * (xs[d+1] - xs[d]**2)**2 + (1.0 - xs[d])**2
                              for d in range(n_dims - 1))
                    return val / n_dims
                elif func_type == "ackley":
                    xs = diff * 4.0
                    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.mean(xs**2)))
                                 - np.exp(np.mean(np.cos(2 * np.pi * xs))) + 20.0 + np.e)
                elif func_type == "rastrigin":
                    xs = diff * 5.12
                    return float(10 * n_dims + np.sum(xs**2 - 10 * np.cos(2 * np.pi * xs))) / n_dims
                return 0.0
            return objective

        instances.append(BenchmarkInstance(
            name=f"synth_{func_type}_{n_dims}d_{i}",
            objective=make_objective(center.copy(), scales.copy(), func_type, n_dims),
            search_space=search_space,
            y_min=0.0,
            y_max=50.0,
            n_dims=n_dims,
            minimize=True,
            source="synthetic",
        ))

    return BenchmarkSuite(
        name="synthetic",
        instances=instances,
        description=f"{n_instances} analytical functions (quadratic/rosenbrock/ackley/rastrigin)",
    )


# ============================================================================
# Tier 2: HPO-B tabular benchmark (real ML data, discrete)
# ============================================================================

class _TabularObjective:
    """Honest tabular objective for HPO-B benchmarking.

    HPO-B provides a finite table of (config, value) pairs — there is no
    ground-truth continuous landscape. Rather than hallucinating one via
    nearest-neighbor interpolation, this class implements the benchmark
    as what it actually is: a discrete selection problem.

    At each trial the optimizer proposes a continuous config. We snap it
    to the nearest *unused* table entry (Euclidean in [0,1]^d), return
    that entry's real pre-computed value, and mark it consumed. If all
    entries are consumed, we return the table-worst value.

    This design:
      - Never returns a value that wasn't actually observed in the data
      - Penalizes optimizers that waste budget re-visiting the same region
      - Computes regret against the table's true best (no fictional y_min)
    """

    def __init__(
        self,
        configs: np.ndarray,
        objectives: np.ndarray,
        hp_names: List[str],
        minimize: bool = True,
    ):
        self.configs = configs.copy()  # (N, D), normalized [0,1]
        self.objectives = objectives.copy()  # (N,)
        self.hp_names = hp_names
        self.minimize = minimize

        # Pre-build KD-tree for fast nearest-neighbor lookup
        from scipy.spatial import KDTree
        self.tree = KDTree(self.configs)

        # Track which entries have been consumed (reset per study)
        self._used: Set[int] = set()

    def reset(self):
        """Reset the consumed-entries set. Call before each new study."""
        self._used = set()

    def __call__(self, trial: Any) -> float:
        """Optuna-compatible objective: propose → snap → return real value."""
        query = np.array([
            trial.suggest_float(name, 0.0, 1.0) for name in self.hp_names
        ], dtype=np.float32)

        # Find nearest unused entry
        # Query multiple neighbors in case the closest ones are used
        k = min(len(self.configs), len(self._used) + 10)
        dists, idxs = self.tree.query(query, k=k)

        # Handle single-result case (k=1 returns scalar)
        if np.ndim(idxs) == 0:
            idxs = np.array([int(idxs)])
            dists = np.array([float(dists)])

        chosen_idx = None
        for idx in idxs:
            idx = int(idx)
            if idx not in self._used:
                chosen_idx = idx
                break

        if chosen_idx is None:
            # All table entries exhausted — return worst value as penalty
            if self.minimize:
                return float(np.max(self.objectives))
            else:
                return float(np.min(self.objectives))

        self._used.add(chosen_idx)
        return float(self.objectives[chosen_idx])


def create_hpob_benchmark(
    data_dir: str = "data/hpob",
    max_instances: int = 50,
    seed: int = 42,
    training_manifest: Optional[str] = None,
) -> Optional[BenchmarkSuite]:
    """Create a tabular benchmark suite from HPO-B.

    Uses the meta_test split exclusively for evaluation. If meta_test is
    unavailable, falls back to meta_validation → meta_train (with a
    warning). Skips any instances found in the training manifest.

    Regret is normalized against the table's actual best and worst values
    (no fictional optima). This is methodologically honest: we measure
    how close the optimizer gets to the best entry the table actually
    contains.

    Args:
        data_dir: HPO-B data directory.
        max_instances: Cap on benchmark instances.
        seed: Random seed for subsampling.
        training_manifest: Path to training_manifest.json for leakage
            prevention. If provided, any instance used during training
            is excluded from the benchmark.

    Returns:
        BenchmarkSuite, or None if HPO-B is unavailable.
    """
    try:
        from hpob_handler import HPOBHandler
    except ImportError:
        print("WARNING: hpob-handler not installed. Skipping HPO-B benchmark.")
        return None

    handler = HPOBHandler(root_dir=data_dir, mode="v3")
    rng = np.random.RandomState(seed)

    # Load training manifest for leakage prevention
    train_ids = set()
    if training_manifest:
        train_ids = load_training_manifest(training_manifest)
        if train_ids:
            print(f"  Loaded {len(train_ids)} training instance IDs for leakage exclusion")

    # Prioritize evaluation splits (avoid leakage from meta_train)
    eval_data = None
    eval_split = None
    for split_name, attr_name in[
        ("meta_test", "meta_test_data"),
        ("meta_validation", "meta_validation_data"),
        ("meta_train", "meta_train_data"),
    ]:
        data = getattr(handler, attr_name, None)
        if data:
            eval_data = data
            eval_split = split_name
            break

    if eval_data is None:
        print("WARNING: No HPO-B data found. Skipping.")
        return None

    if eval_split == "meta_train":
        print("  WARNING: Using meta_train for benchmarking (no test split available). "
              "Risk of data leakage if training also used meta_train.")

    print(f"  HPO-B eval split: {eval_split}")

    instances =[]
    search_spaces = handler.get_search_spaces()

    for ss_id in search_spaces:
        if ss_id not in eval_data:
            continue
        for ds_id in eval_data[ss_id]:
            # Leakage check
            leak_key = f"hpob/{eval_split}/{ss_id}/{ds_id}"
            if (
                leak_key in train_ids
                or f"hpob/train/{ss_id}/{ds_id}" in train_ids
                or f"hpob/init/{ss_id}/{ds_id}" in train_ids
            ):
                continue

            try:
                data_dict = eval_data[ss_id][ds_id]
                X, y = data_dict["X"], data_dict["y"]
                configs = np.array(X, dtype=np.float32)
                objectives = np.array(y, dtype=np.float32).flatten()

                if len(configs) < 30:
                    continue

                # Normalize configs to [0, 1]
                c_min = configs.min(axis=0)
                c_max = configs.max(axis=0)
                valid = (c_max - c_min) > 1e-8
                configs_norm = configs.copy()
                configs_norm[:, valid] = (
                    (configs_norm[:, valid] - c_min[valid]) /
                    (c_max[valid] - c_min[valid])
                )
                configs_norm[:, ~valid] = 0.5

                n_dims = configs_norm.shape[1]
                hp_names = [f"x{d}" for d in range(n_dims)]
                search_space = {
                    name: optuna.distributions.FloatDistribution(0.0, 1.0)
                    for name in hp_names
                }

                # Regret bounds: the TABLE's actual best and worst.
                # HPO-B responses are accuracy-style scores, so the benchmark
                # is a maximization problem.
                y_best = float(np.max(objectives))
                y_worst = float(np.min(objectives))
                if abs(y_worst - y_best) < 1e-8:
                    continue

                tabular_obj = _TabularObjective(
                    configs_norm, objectives, hp_names, minimize=False,
                )

                instances.append(BenchmarkInstance(
                    name=f"hpob_{ss_id}_{ds_id}",
                    objective=tabular_obj,
                    search_space=search_space,
                    y_min=y_worst,
                    y_max=y_best,
                    n_dims=n_dims,
                    minimize=False,
                    source="hpob",
                    metadata={
                        "search_space": ss_id,
                        "dataset": ds_id,
                        "split": eval_split,
                        "table_size": len(configs),
                    },
                ))
            except Exception as e:
                print(f"    Skipping HPO-B {ss_id}/{ds_id}: {e}")

    if not instances:
        print("WARNING: No valid HPO-B instances found.")
        return None

    if len(instances) > max_instances:
        indices = rng.choice(len(instances), max_instances, replace=False)
        instances = [instances[i] for i in indices]

    table_sizes = [inst.metadata["table_size"] for inst in instances]
    print(f"  HPO-B benchmark: {len(instances)} instances, "
          f"table sizes {np.min(table_sizes)}-{np.max(table_sizes)} "
          f"(median {np.median(table_sizes):.0f})")

    return BenchmarkSuite(
        name="hpob",
        instances=instances,
        description=f"HPO-B tabular benchmark ({eval_split} split), "
                    f"{len(instances)} instances",
    )


# ============================================================================
# Tier 1: YAHPO Gym continuous benchmark (real ML/DL data)
# ============================================================================

def create_yahpo_benchmark(
    max_instances_per_scenario: int = 10,
    scenarios: Optional[List[str]] = None,
    seed: int = 42,
    training_manifest: Optional[str] = None,
) -> Optional[BenchmarkSuite]:
    """Create a continuous benchmark suite from YAHPO Gym surrogates.

    YAHPO provides fitted surrogate models (ONNX random forests) that
    accept arbitrary configurations and return smooth predictions. This
    is a legitimate continuous benchmark — no discretization artifacts.

    Args:
        max_instances_per_scenario: Max instances per scenario.
        scenarios: YAHPO scenarios to use. None = curated default set.
        seed: Random seed.
        training_manifest: Path to training_manifest.json for leakage
            prevention.

    Returns:
        BenchmarkSuite, or None if YAHPO is unavailable.
    """
    try:
        with redirect_stdout(io.StringIO()):
            from yahpo_gym import BenchmarkSet, local_config
    except ImportError:
        print("WARNING: yahpo_gym not installed. Skipping YAHPO benchmark.")
        return None

    if scenarios is None:
        scenarios =["lcbench", "rbv2_svm", "rbv2_ranger", "rbv2_xgboost"]

    yahpo_data_path = getattr(local_config, "data_path", None)
    yahpo_root = Path(yahpo_data_path) if yahpo_data_path else None
    if (
        yahpo_root is None
        or not yahpo_root.exists()
        or not _has_yahpo_scenario_assets(yahpo_root, scenarios)
    ):
        print(
            "WARNING: YAHPO data path is not configured correctly: "
            f"{yahpo_data_path!r}"
        )
        return None

    rng = np.random.RandomState(seed)

    # Load training manifest for leakage prevention
    train_ids = set()
    if training_manifest:
        train_ids = load_training_manifest(training_manifest)

    instances =[]

    for scenario_name in scenarios:
        try:
            with redirect_stdout(io.StringIO()):
                bench = BenchmarkSet(scenario_name)
        except Exception as e:
            print(f"  Skipping YAHPO scenario {scenario_name}: {e}")
            continue

        # Identify target metric and direction
        target_metric, minimize = _yahpo_target_metric(bench)
        if target_metric is None:
            print(f"  Skipping {scenario_name}: no suitable target metric")
            continue

        # Select instances, excluding any used in training
        all_inst = list(bench.instances)
        eligible =[]
        for inst_id in all_inst:
            leak_key = f"yahpo/{scenario_name}/{inst_id}"
            if leak_key not in train_ids:
                eligible.append(inst_id)

        if not eligible:
            print(f"  Skipping {scenario_name}: all instances used in training")
            continue

        if len(eligible) > max_instances_per_scenario:
            selected = list(rng.choice(eligible, max_instances_per_scenario, replace=False))
        else:
            selected = eligible

        print(f"  YAHPO {scenario_name}: {len(selected)} eval instances "
              f"(excluded {len(all_inst) - len(eligible)} training instances) | "
              f"metric={target_metric} | minimize={minimize}")

        for inst_id in selected:
            try:
                inst = _create_yahpo_instance(
                    scenario_name, inst_id, target_metric, minimize,
                )
                if inst is not None:
                    instances.append(inst)
            except Exception as e:
                print(f"    Skipping {scenario_name}/{inst_id}: {e}")

    if not instances:
        print("WARNING: No valid YAHPO instances found.")
        return None

    print(f"  YAHPO benchmark total: {len(instances)} instances "
          f"across {len(scenarios)} scenarios")

    return BenchmarkSuite(
        name="yahpo",
        instances=instances,
        description=f"YAHPO Gym continuous surrogate benchmark, "
                    f"{len(instances)} instances",
    )


def _yahpo_target_metric(bench: Any) -> Tuple[Optional[str], bool]:
    """Determine the target metric and optimization direction for a YAHPO scenario."""
    # Ordered by preference: accuracy-like (maximize) then loss-like (minimize)
    candidates =[
        ("val_accuracy", False),
        ("val_balanced_accuracy", False),
        ("acc", False),
        ("auc", False),
        ("f1", False),
        ("logloss", True),
        ("mmce", True),
        ("nf", True),
        ("time_train", True),
    ]
    targets = getattr(bench, 'targets',[]) or[]
    for metric, is_minimize in candidates:
        if metric in targets:
            return metric, is_minimize
    if targets:
        return targets[0], True  # fallback: assume minimize
    return None, True


def _yahpo_target_bounds(bench: Any, target_metric: str) -> Optional[Tuple[float, float]]:
    """Extract empirical target bounds from YAHPO target_stats if available."""
    stats = getattr(bench, "target_stats", None)
    if stats is None:
        return None

    records = None
    if hasattr(stats, "to_dict"):
        try:
            records = stats.to_dict("records")
        except TypeError:
            try:
                records = stats.to_dict(orient="records")
            except Exception:
                records = None
    if records is None and hasattr(stats, "iterrows"):
        records =[row.to_dict() for _, row in stats.iterrows()]
    if records is None and isinstance(stats, list):
        records = stats
    if not records:
        return None

    metric_rows =[
        row for row in records
        if str(row.get("metric", "")) == target_metric
    ]
    if not metric_rows:
        return None

    bounds = {}
    for row in metric_rows:
        stat = str(row.get("statistic", "")).lower()
        value = row.get("value")
        if stat in {"min", "max"} and value is not None:
            bounds[stat] = float(value)

    if "min" not in bounds or "max" not in bounds:
        return None
    if abs(bounds["max"] - bounds["min"]) < 1e-8:
        return None
    return bounds["min"], bounds["max"]


def _has_yahpo_scenario_assets(data_root: Path, scenarios: List[str]) -> bool:
    """Check whether a YAHPO data root contains at least one scenario payload."""
    return any((data_root / scenario_name / "encoding.json").exists() for scenario_name in scenarios)


def _create_yahpo_instance(
    scenario_name: str,
    inst_id: Any,
    target_metric: str,
    minimize: bool,
) -> Optional[BenchmarkInstance]:
    """Create a single YAHPO benchmark instance.

    Uses YAHPO's empirical target statistics for regret normalization when
    available, falling back to local random probing otherwise, then builds
    a fresh BenchmarkSet closure for the actual objective.
    """
    with redirect_stdout(io.StringIO()):
        from yahpo_gym import BenchmarkSet
        # Prefer the official target statistics exposed by YAHPO. Fall back to
        # local probing only when those statistics are unavailable.
        probe = BenchmarkSet(scenario_name)
        probe.set_instance(str(inst_id))
        
    cs = probe.get_opt_space()
    target_bounds = _yahpo_target_bounds(probe, target_metric)
    normalization_source = "target_stats"
    n_probe_samples = 0

    if target_bounds is not None:
        y_min, y_max = target_bounds
    else:
        normalization_source = "random_probe"
        pre_configs = cs.sample_configuration(300)
        if not isinstance(pre_configs, list):
            pre_configs = [pre_configs]

        pre_config_dicts =[
            cfg.get_dictionary() if hasattr(cfg, "get_dictionary") else dict(cfg)
            for cfg in pre_configs
        ]

        # --- ADD THIS BLOCK ---
        for cfg in pre_config_dicts:
            if hasattr(probe, 'config_space') and 'task_id' in probe.config_space:
                if 'task_id' not in cfg:
                    cfg['task_id'] = probe.instances[0] if hasattr(probe, 'instances') and len(probe.instances) > 0 else (probe.active_session if hasattr(probe, 'active_session') else str(inst_id))
            if hasattr(probe, 'config_space') and 'OpenML_task_id' in probe.config_space and 'OpenML_task_id' not in cfg:
                cfg['OpenML_task_id'] = probe.instances[0] if hasattr(probe, 'instances') and len(probe.instances) > 0 else (probe.active_session if hasattr(probe, 'active_session') else str(inst_id))
        # ----------------------

        # YAHPO supports batched objective queries; fall back to per-config
        # evaluation only if a scenario-specific wrapper rejects the batch.
        try:
            probe_results = probe.objective_function(pre_config_dicts)
            if not isinstance(probe_results, list):
                probe_results = [probe_results]
        except Exception:
            probe_results =[]
            for cfg in pre_config_dicts:
                try:
                    res = probe.objective_function([cfg])
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]
                    probe_results.append(res)
                except Exception:
                    probe_results.append(None)

        vals =[]
        for res in probe_results:
            if res is None:
                continue
            v = res.get(target_metric, None)
            if v is not None and not np.isnan(v):
                vals.append(float(v))
        if len(vals) < 20:
            return None

        n_probe_samples = len(vals)
        if minimize:
            y_min = float(np.percentile(vals, 1))   # near-best
            y_max = float(np.percentile(vals, 95))  # near-worst
        else:
            y_min = float(np.percentile(vals, 5))   # near-worst
            y_max = float(np.percentile(vals, 99))  # near-best
        if abs(y_max - y_min) < 1e-8:
            return None

    # Build Optuna search space from ConfigSpace
    search_space = _configspace_to_optuna(cs)
    n_dims = len(search_space)

    # Objective closure — each call creates a fresh BenchmarkSet to avoid
    # shared state between seeds
    def make_objective(scen, inst, metric, cs_template):
        def objective(trial):
            with redirect_stdout(io.StringIO()):
                from yahpo_gym import BenchmarkSet
                b = BenchmarkSet(scen)
                b.set_instance(str(inst))
                
            local_cs = b.get_opt_space()
            cfg = _sample_from_trial(trial, local_cs)

            # --- ADD THIS BLOCK ---
            # YAHPO Gym requires the task_id to be present in the configuration dictionary
            # for multi-task scenarios when converting it back to a Configuration object.
            if hasattr(b, 'config_space') and 'task_id' in b.config_space:
                if 'task_id' not in cfg:
                    # Inject the active instance ID back into the dictionary
                    cfg['task_id'] = b.instances[0] if hasattr(b, 'instances') and len(b.instances) > 0 else (b.active_session if hasattr(b, 'active_session') else str(inst))
            # ----------------------
            if hasattr(b, 'config_space') and 'OpenML_task_id' in b.config_space and 'OpenML_task_id' not in cfg:
                cfg['OpenML_task_id'] = b.instances[0] if hasattr(b, 'instances') and len(b.instances) > 0 else (b.active_session if hasattr(b, 'active_session') else str(inst))

            # Single-config calls are valid; the batched API is used only where
            # it materially reduces overhead.
            result = b.objective_function([cfg])
            
            # Since we passed a list of one config, the result should be a list of one result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if result is None:
                return float("inf") if minimize else float("-inf")
            val = result.get(metric, None)
            if val is None or np.isnan(val):
                return float("inf") if minimize else float("-inf")
            return float(val)
        return objective

    return BenchmarkInstance(
        name=f"yahpo_{scenario_name}_{inst_id}",
        objective=make_objective(scenario_name, inst_id, target_metric, cs),
        search_space=search_space,
        y_min=y_min,
        y_max=y_max,
        n_dims=n_dims,
        minimize=minimize,
        source="yahpo",
        metadata={
            "scenario": scenario_name,
            "instance_id": str(inst_id),
            "target_metric": target_metric,
            "normalization_source": normalization_source,
            "n_probe_samples": n_probe_samples,
        },
    )


def _configspace_to_optuna(cs: Any) -> Dict[str, Any]:
    """Convert a ConfigSpace to Optuna distributions."""
    search_space = {}
    for hp in cs.get_hyperparameters():
        if hasattr(hp, 'choices'):
            search_space[hp.name] = optuna.distributions.CategoricalDistribution(
                list(hp.choices)
            )
        elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            if isinstance(hp.lower, float):
                search_space[hp.name] = optuna.distributions.FloatDistribution(
                    float(hp.lower), float(hp.upper),
                    log=getattr(hp, 'log', False),
                )
            else:
                search_space[hp.name] = optuna.distributions.IntDistribution(
                    int(hp.lower), int(hp.upper),
                    log=getattr(hp, 'log', False),
                )
    return search_space


def _sample_from_trial(trial: Any, cs: Any) -> Dict[str, Any]:
    """Use Optuna trial to suggest values matching a ConfigSpace."""
    cfg = {}
    for hp in cs.get_hyperparameters():
        if hasattr(hp, 'choices'):
            cfg[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
        elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            if isinstance(hp.lower, float):
                cfg[hp.name] = trial.suggest_float(
                    hp.name, float(hp.lower), float(hp.upper),
                    log=getattr(hp, 'log', False),
                )
            else:
                cfg[hp.name] = trial.suggest_int(
                    hp.name, int(hp.lower), int(hp.upper),
                    log=getattr(hp, 'log', False),
                )

    # Clean inactive hyperparameters using ConfigSpace validation
    if hasattr(cs, "get_conditions") and len(cs.get_conditions()) > 0:
        try:
            import ConfigSpace as CS
            import re
            while True:
                try:
                    # CS.Configuration initialization will strip out inactive parameters
                    # if they aren't strictly validated, or it will throw an exception
                    clean_config = CS.Configuration(cs, values=cfg)
                    cfg = clean_config.get_dictionary()
                    break
                except Exception as e:
                    msg = str(e)
                    # ConfigSpace raises: ValueError: Inactive hyperparameter 'degree' must not be specified...
                    match = re.search(r"Inactive hyperparameter '([^']+)'", msg)
                    if match:
                        hp_name = match.group(1)
                        if hp_name in cfg:
                            del cfg[hp_name]
                        else:
                            break
                    else:
                        raise e
        except ImportError:
            pass

    return cfg


# ============================================================================
# Benchmark execution engine
# ============================================================================

def _create_sampler(method: str, seed: int, model_path: Optional[str] = None) -> Any:
    """Create an Optuna sampler for a given method."""
    if method == "trip_tpe":
        from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler
        return TRIPTPESampler(
            model_path=model_path, seed=seed,
            n_warmup_trials=5, requery_interval=10,
        )
    elif method == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    elif method == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif method == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:
        return optuna.samplers.TPESampler(seed=seed)


def run_single_benchmark(
    method: str,
    instance: BenchmarkInstance,
    n_trials: int,
    seed: int,
    model_path: Optional[str] = None,
) -> BenchmarkResult:
    """Run a single optimization benchmark.

    For tabular benchmarks (HPO-B), resets the objective's consumed-entries
    set before each run so every seed starts fresh.
    """
    # Reset tabular state if applicable
    if isinstance(instance.objective, _TabularObjective):
        instance.objective.reset()

    direction = "minimize" if instance.minimize else "maximize"
    sampler = _create_sampler(method, seed, model_path)
    study = optuna.create_study(direction=direction, sampler=sampler)

    t0 = time.time()
    study.optimize(instance.objective, n_trials=n_trials, show_progress_bar=False)
    wall_time = time.time() - t0

    trajectory = np.array([t.value for t in study.trials if t.value is not None])

    regret_at_budgets = {}
    for budget in[25, 50, 100]:
        if budget <= len(trajectory):
            regret_at_budgets[budget] = normalized_regret_at_budget(
                trajectory, instance.y_min, instance.y_max, budget,
                minimize=instance.minimize,
            )

    aurc = area_under_regret_curve(
        trajectory, instance.y_min, instance.y_max,
        min(100, len(trajectory)), minimize=instance.minimize,
    )

    return BenchmarkResult(
        method=method,
        benchmark=instance.source,
        instance=instance.name,
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
    """Run a full benchmark suite across methods and seeds."""
    results: Dict[str, List[BenchmarkResult]] = {m:[] for m in methods}

    total_runs = len(methods) * len(suite.instances) * n_seeds
    print(f"\nBenchmark: {suite.name} ({suite.description})")
    print(f"Instances: {len(suite.instances)}, Methods: {len(methods)}, Seeds: {n_seeds}")
    print(f"Total runs: {total_runs}")
    print("-" * 60)

    run_idx = 0
    for instance in suite.instances:
        for method in methods:
            for seed_offset in range(n_seeds):
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] {method} | {instance.name} | seed={seed_offset}",
                    end=" ... ", flush=True,
                )

                result = run_single_benchmark(
                    method=method, instance=instance,
                    n_trials=n_trials, seed=seed_offset,
                    model_path=model_path,
                )
                results[method].append(result)
                print(f"best={result.best_value:.4f} | time={result.wall_time:.1f}s")

                if use_wandb:
                    _log_wandb_run(suite.name, method, result, instance)

    _print_summary(suite.name, methods, results)
    _save_results(suite, methods, results, output_dir)

    if use_wandb:
        _log_wandb_suite_summary(suite, methods, results, output_dir, n_trials)

    return results


# ============================================================================
# Reporting
# ============================================================================

def _print_summary(suite_name: str, methods: List[str], results: Dict[str, List[BenchmarkResult]]):
    """Print text summary of benchmark results."""
    print("\n" + "=" * 60)
    print(f"SUMMARY: {suite_name}")
    print("=" * 60)

    for budget in[25, 50, 100]:
        print(f"\n--- Normalized Regret @ Budget={budget} ---")
        regret_by_method = {}
        for method in methods:
            regrets =[r.regret_at_budgets.get(budget, float("nan"))
                       for r in results[method] if budget in r.regret_at_budgets]
            if regrets:
                regret_by_method[method] = regrets
                print(f"  {method:15s}: {np.nanmean(regrets):.4f} +/- {np.nanstd(regrets):.4f}")

        if len(regret_by_method) > 1:
            min_len = min(len(v) for v in regret_by_method.values())
            truncated = {k: v[:min_len] for k, v in regret_by_method.items()}
            ranks = average_rank(truncated)
            print(f"\n  Average Rank:")
            for method, rank in sorted(ranks.items(), key=lambda x: x[1]):
                print(f"    {method:15s}: {rank:.2f}")

    if "trip_tpe" in methods and len(methods) > 1:
        print(f"\n--- Wilcoxon Signed-Rank Tests (TRIP-TPE vs baselines) ---")
        trip_aurcs = [r.aurc for r in results["trip_tpe"]]
        for method in methods:
            if method == "trip_tpe":
                continue
            base_aurcs = [r.aurc for r in results[method]]
            n = min(len(trip_aurcs), len(base_aurcs))
            if n >= 5:
                _, pval = wilcoxon_signed_rank_test(trip_aurcs[:n], base_aurcs[:n])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  TRIP-TPE vs {method:15s}: p={pval:.4f} {sig}")


def _save_results(
    suite: BenchmarkSuite,
    methods: List[str],
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
):
    """Save results to JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for method in methods:
        method_results = results[method]
        summary[method] = {
            "mean_aurc": float(np.mean([r.aurc for r in method_results])),
            "mean_best": float(np.mean([r.best_value for r in method_results])),
            "mean_wall_time": float(np.mean([r.wall_time for r in method_results])),
            "n_runs": len(method_results),
            "regret_at_budgets": {
                str(b): float(np.nanmean([
                    r.regret_at_budgets.get(b, float("nan"))
                    for r in method_results
                ]))
                for b in [25, 50, 100]
            },
        }

    # Also save per-instance detail
    detail =[]
    for method in methods:
        for r in results[method]:
            detail.append({
                "method": r.method,
                "instance": r.instance,
                "seed": r.seed,
                "best_value": r.best_value,
                "aurc": r.aurc,
                "wall_time": r.wall_time,
                "regret_at_budgets": r.regret_at_budgets,
            })

    output = {
        "suite": suite.name,
        "description": suite.description,
        "summary": summary,
        "detail": detail,
    }

    path = out_dir / f"{suite.name}_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


# ============================================================================
# W&B logging
# ============================================================================

def _log_wandb_run(suite_name: str, method: str, result: BenchmarkResult, instance: BenchmarkInstance):
    """Log a single benchmark run to W&B."""
    log_dict = {
        f"bench/{suite_name}/{method}/best_value": result.best_value,
        f"bench/{suite_name}/{method}/wall_time": result.wall_time,
        f"bench/{suite_name}/{method}/aurc": result.aurc,
    }
    for b, r in result.regret_at_budgets.items():
        log_dict[f"bench/{suite_name}/{method}/regret@{b}"] = r
    wandb.log(log_dict)


def _log_wandb_suite_summary(
    suite: BenchmarkSuite,
    methods: List[str],
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
    n_trials: int,
):
    """Log comprehensive W&B summary for a benchmark suite."""

    # --- Summary comparison table ---
    table = wandb.Table(
        columns=["suite", "method", "mean_aurc", "std_aurc", "mean_best",
                 "mean_wall_time", "regret@25", "regret@50", "regret@100", "n_runs"]
    )
    for method in methods:
        aurcs = [r.aurc for r in results[method]]
        bests = [r.best_value for r in results[method]]
        times =[r.wall_time for r in results[method]]
        r25 =[r.regret_at_budgets.get(25, float("nan")) for r in results[method]]
        r50 =[r.regret_at_budgets.get(50, float("nan")) for r in results[method]]
        r100 =[r.regret_at_budgets.get(100, float("nan")) for r in results[method]]
        table.add_data(
            suite.name, method,
            float(np.mean(aurcs)), float(np.std(aurcs)),
            float(np.mean(bests)), float(np.mean(times)),
            float(np.nanmean(r25)), float(np.nanmean(r50)), float(np.nanmean(r100)),
            len(aurcs),
        )
    wandb.log({f"benchmark/{suite.name}/summary": table})

    # --- Convergence curves per method ---
    for method in methods:
        trajs = [r.trajectory for r in results[method]]
        if not trajs:
            continue
        # Use median y_min/y_max across instances for normalization
        y_mins = [inst.y_min for inst in suite.instances]
        y_maxs = [inst.y_max for inst in suite.instances]
        y_min = float(np.median(y_mins))
        y_max = float(np.median(y_maxs))
        mean_reg, lo_ci, hi_ci = convergence_curve(trajs, y_min, y_max, n_trials)
        data = [[i, float(mean_reg[i]), float(lo_ci[i]), float(hi_ci[i])]
                for i in range(len(mean_reg))]
        conv_table = wandb.Table(
            data=data, columns=["trial", "mean_regret", "ci_lower", "ci_upper"]
        )
        wandb.log({
            f"benchmark/{suite.name}/convergence/{method}": wandb.plot.line(
                conv_table, "trial", "mean_regret",
                title=f"{suite.name}: {method}"
            )
        })

    # --- Regret bar charts per budget ---
    for budget in [25, 50, 100]:
        bar_data =[]
        for method in methods:
            regrets =[r.regret_at_budgets.get(budget, float("nan"))
                       for r in results[method] if budget in r.regret_at_budgets]
            if regrets:
                bar_data.append([method, float(np.nanmean(regrets)), float(np.nanstd(regrets))])
        if bar_data:
            bar_table = wandb.Table(data=bar_data, columns=["method", "mean_regret", "std_regret"])
            wandb.log({
                f"benchmark/{suite.name}/regret_bar@{budget}": wandb.plot.bar(
                    bar_table, "method", "mean_regret",
                    title=f"{suite.name}: Regret @ {budget}"
                )
            })

    # --- Per-instance detail table ---
    detail_rows =[]
    for method in methods:
        for r in results[method]:
            detail_rows.append([
                suite.name, method, r.instance, r.seed,
                r.best_value, r.wall_time, r.aurc,
                r.regret_at_budgets.get(25, float("nan")),
                r.regret_at_budgets.get(50, float("nan")),
                r.regret_at_budgets.get(100, float("nan")),
            ])
    detail_table = wandb.Table(
        data=detail_rows,
        columns=["suite", "method", "instance", "seed",
                 "best_value", "wall_time", "aurc",
                 "regret@25", "regret@50", "regret@100"]
    )
    wandb.log({f"benchmark/{suite.name}/detailed_results": detail_table})

    # --- Results artifact ---
    results_path = Path(output_dir) / f"{suite.name}_results.json"
    if results_path.exists():
        artifact = wandb.Artifact(
            f"trip-tpe-bench-{suite.name}", type="results",
            description=f"{suite.name}: {len(methods)} methods, "
                        f"{len(suite.instances)} instances",
        )
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)


# ============================================================================
# Multi-suite orchestrator
# ============================================================================

def run_all_benchmarks(
    benchmark_types: List[str],
    methods: List[str],
    n_trials: int = 100,
    n_seeds: int = 5,
    model_path: Optional[str] = None,
    output_dir: str = "results",
    use_wandb: bool = False,
    hpob_dir: str = "data/hpob",
    n_synthetic_instances: int = 10,
    yahpo_scenarios: Optional[List[str]] = None,
    training_manifest: Optional[str] = None,
) -> Dict[str, Dict[str, List[BenchmarkResult]]]:
    """Run benchmarks across multiple suites and aggregate.

    Args:
        benchmark_types: List of "synthetic", "hpob", "yahpo".
        methods: Methods to compare.
        n_trials: Budget per run.
        n_seeds: Seeds per (method, instance).
        model_path: TRIP-TPE model checkpoint.
        output_dir: Output directory.
        use_wandb: Enable W&B.
        hpob_dir: HPO-B data directory.
        n_synthetic_instances: Synthetic benchmark instances.
        yahpo_scenarios: YAHPO scenarios.
        training_manifest: Path to training_manifest.json for leakage
            prevention.

    Returns:
        {suite_name: {method: [BenchmarkResult, ...]}}.
    """
    all_results = {}

    for btype in benchmark_types:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK SUITE: {btype.upper()}")
        print(f"{'=' * 60}")

        suite = None
        if btype == "synthetic":
            suite = create_synthetic_benchmark(n_instances=n_synthetic_instances)
        elif btype == "hpob":
            suite = create_hpob_benchmark(
                data_dir=hpob_dir,
                training_manifest=training_manifest,
            )
        elif btype == "yahpo":
            suite = create_yahpo_benchmark(
                scenarios=yahpo_scenarios,
                training_manifest=training_manifest,
            )
        else:
            print(f"Unknown benchmark type: {btype}")
            continue

        if suite is None:
            print(f"  Skipping {btype} (not available)")
            continue

        results = run_benchmark_suite(
            suite=suite, methods=methods,
            n_trials=n_trials, n_seeds=n_seeds,
            model_path=model_path, output_dir=output_dir,
            use_wandb=use_wandb,
        )
        all_results[suite.name] = results

    # Cross-suite aggregation
    if len(all_results) > 1:
        _print_cross_suite_summary(all_results, methods)
        if use_wandb:
            _log_wandb_cross_suite(all_results, methods)

    return all_results


def _print_cross_suite_summary(
    all_results: Dict[str, Dict[str, List[BenchmarkResult]]],
    methods: List[str],
):
    """Print aggregated results across all benchmark suites."""
    print(f"\n{'=' * 60}")
    print("CROSS-SUITE AGGREGATION")
    print(f"{'=' * 60}")

    overall = {m:[] for m in methods}
    for suite_name, suite_results in all_results.items():
        print(f"\n  Suite: {suite_name}")
        for method in methods:
            if method not in suite_results:
                continue
            aurcs =[r.aurc for r in suite_results[method]]
            overall[method].extend(aurcs)
            print(f"    {method:15s}: AURC = {np.mean(aurcs):.4f} +/- {np.std(aurcs):.4f}")

    print(f"\n  Overall (all suites combined):")
    for method in sorted(methods, key=lambda m: np.mean(overall.get(m, [1.0]))):
        if overall[method]:
            print(f"    {method:15s}: AURC = {np.mean(overall[method]):.4f}")


def _log_wandb_cross_suite(
    all_results: Dict[str, Dict[str, List[BenchmarkResult]]],
    methods: List[str],
):
    """Log cross-suite W&B comparison."""
    rows =[]
    for suite_name, suite_results in all_results.items():
        for method in methods:
            if method not in suite_results:
                continue
            aurcs = [r.aurc for r in suite_results[method]]
            rows.append([suite_name, method, float(np.mean(aurcs)),
                         float(np.std(aurcs)), len(aurcs)])
    if rows:
        table = wandb.Table(
            data=rows, columns=["suite", "method", "mean_aurc", "std_aurc", "n_runs"]
        )
        wandb.log({"benchmark/cross_suite_comparison": table})

    # Overall ranking
    overall = {m:[] for m in methods}
    for suite_results in all_results.values():
        for method in methods:
            if method in suite_results:
                overall[method].extend([r.aurc for r in suite_results[method]])
    if all(len(v) > 0 for v in overall.values()):
        wandb.run.summary["overall_ranking"] = dict(
            sorted({m: float(np.mean(v)) for m, v in overall.items()}.items(),
                   key=lambda x: x[1])
        )


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark TRIP-TPE against baselines")
    parser.add_argument("--model-path", type=str, default=None, help="Path to TRIP-TPE model")
    parser.add_argument(
        "--methods", nargs="+", default=["trip_tpe", "tpe", "random"],
        help="Methods to benchmark",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["yahpo", "hpob", "synthetic"],
        choices=["synthetic", "hpob", "yahpo"],
        help="Benchmark suites (default: yahpo hpob synthetic)",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Budget per run")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds per (method, instance)")
    parser.add_argument("--n-instances", type=int, default=10, help="Synthetic instances")
    parser.add_argument("--hpob-dir", type=str, default="data/hpob", help="HPO-B data dir")
    parser.add_argument("--output-dir", type=str, default="results", help="Output dir")
    parser.add_argument(
        "--training-manifest", type=str, default=None,
        help="Path to training_manifest.json for data leakage prevention",
    )
    parser.add_argument("--wandb-project", type=str, default="trip-tpe", help="W&B project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                job_type="benchmark",
                config=vars(args),
                tags=["benchmark"] +[f"suite-{b}" for b in args.benchmarks],
            )
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging.")
            use_wandb = False

    run_all_benchmarks(
        benchmark_types=args.benchmarks,
        methods=args.methods,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=use_wandb,
        hpob_dir=args.hpob_dir,
        n_synthetic_instances=args.n_instances,
        training_manifest=args.training_manifest,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()