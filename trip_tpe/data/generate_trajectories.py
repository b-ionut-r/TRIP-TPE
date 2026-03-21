"""
Trajectory generation script for TRIP-TPE.

Generates training data from surrogate benchmarks (HPO-B, YAHPO Gym)
or synthetic functions. All trajectory generation is CPU-bound,
preserving GPU resources exclusively for Transformer training.

Usage:
    trip-tpe-generate --mode synthetic --n-trajectories 50000 --output data/train.pt
    trip-tpe-generate --mode hpob --output data/hpob_train.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from trip_tpe.data.preprocessing import META_FEATURE_DIM, TrajectoryPreprocessor, TrajectoryPair
from trip_tpe.data.trajectory_dataset import TrajectoryDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _generate_synthetic_meta_features(rng: np.random.RandomState) -> np.ndarray:
    """Generate plausible synthetic dataset meta-features for training.

    Produces realistic-looking Tier 1 + Tier 2 meta-features that mimic
    the distribution of real ML datasets, ensuring the Transformer learns
    to condition on meta-features during pre-training.

    Returns:
        Meta-features array, shape (META_FEATURE_DIM,), all values in [0, 1].
        Slots 0-6: Tier 1 (basic properties)
        Slots 7-9: Tier 2 (aggregate statistics)
    """
    # ---- Tier 1 (slots 0-6) ----

    # log10(n_samples) / 6 — datasets range from 100 to 1M samples
    log_n_samples = rng.uniform(2.0, 6.0) / 6.0  # [0.33, 1.0]

    # log10(n_features) / 4 — features range from 2 to 10000
    log_n_features = rng.uniform(0.3, 4.0) / 4.0  # [0.075, 1.0]

    # n_classes / 20 — binary (2) to many-class (20+)
    n_classes = rng.choice([2, 3, 5, 10, 15, 20]) / 20.0

    # dimensionality ratio (n_features / n_samples), capped at 1.0
    dim_ratio = min(1.0, rng.uniform(0.001, 0.5))

    # imbalance ratio (minority / majority class fraction)
    imbalance = rng.uniform(0.1, 1.0)

    # fraction of categorical features
    frac_cat = rng.uniform(0.0, 1.0)

    # fraction of missing values
    frac_missing = rng.exponential(0.05)  # mostly low, occasionally high
    frac_missing = min(frac_missing, 1.0)

    # ---- Tier 2 (slots 7-9) ----

    # Intrinsic dimensionality ratio: low for tabular with redundant features,
    # high for sparse/independent features. Beta distribution mimics real data.
    intrinsic_dim_ratio = rng.beta(2, 5)  # skewed low (most datasets are redundant)

    # Mean mutual information (normalized): signal-to-noise. Uniform-ish.
    mean_mi = rng.beta(2, 3)  # slightly skewed toward lower signal

    # Landmark 1-NN accuracy: dataset difficulty. Beta-distributed.
    landmark_1nn = rng.beta(3, 2)  # skewed toward easier datasets (realistic)

    return np.array([
        log_n_samples, log_n_features, n_classes,
        dim_ratio, imbalance, frac_cat, frac_missing,
        intrinsic_dim_ratio, mean_mi, landmark_1nn,
    ], dtype=np.float32)


def generate_synthetic_trajectories(
    n_trajectories: int = 50000,
    n_dims_range: tuple = (3, 16),
    n_trials_range: tuple = (30, 200),
    gamma: float = 0.15,
    seed: int = 42,
    include_meta_features: bool = True,
) -> List[TrajectoryPair]:
    """Generate synthetic HPO trajectories from diverse analytical functions.

    Uses a mixture of objective function families to simulate the diversity
    of real HPO landscapes:
        - Quadratic bowls (convex, unimodal)
        - Rosenbrock-like valleys (narrow ridges)
        - Ackley-like functions (multi-modal with global structure)
        - Rastrigin-like functions (highly multi-modal)

    All computation is CPU-only.

    Args:
        n_trajectories: Total trajectories to generate.
        n_dims_range: (min_dims, max_dims) for random dimensionality.
        n_trials_range: (min_trials, max_trials) per trajectory.
        gamma: Quantile for "good" region definition.
        seed: Random seed.
        include_meta_features: If True, generate synthetic dataset-level
            meta-features for each trajectory (for cold-start conditioning).

    Returns:
        List of TrajectoryPair instances.
    """
    rng = np.random.RandomState(seed)
    preprocessor = TrajectoryPreprocessor(
        gamma=gamma,
        min_prefix_len=5,
        max_prefix_len=50,
        n_prefixes_per_trajectory=5,
        seed=seed,
    )

    all_pairs: List[TrajectoryPair] = []
    func_types = ["quadratic", "rosenbrock", "ackley", "rastrigin"]

    for i in tqdm(range(n_trajectories), desc="Generating trajectories"):
        n_dims = rng.randint(n_dims_range[0], n_dims_range[1] + 1)
        n_trials = rng.randint(n_trials_range[0], n_trials_range[1] + 1)
        func_type = func_types[i % len(func_types)]

        # Random center for the optimum
        center = rng.uniform(0.1, 0.9, size=n_dims).astype(np.float32)

        # Generate random configurations in [0, 1]^d
        configs = rng.uniform(0.0, 1.0, size=(n_trials, n_dims)).astype(np.float32)

        # Compute objectives based on function type
        objectives = _compute_objectives(configs, center, func_type, rng, n_dims)

        # Generate synthetic meta-features for cold-start conditioning
        meta = _generate_synthetic_meta_features(rng) if include_meta_features else None

        pairs = preprocessor.process_trajectory(
            configs=configs,
            objectives=objectives,
            search_space_id=f"{func_type}_{n_dims}d_{i}",
            minimize=True,
            meta_features=meta,
        )
        all_pairs.extend(pairs)

    return all_pairs


def _compute_objectives(
    configs: np.ndarray,
    center: np.ndarray,
    func_type: str,
    rng: np.random.RandomState,
    n_dims: int,
) -> np.ndarray:
    """Compute objective values for a given function type.

    Args:
        configs: Configs in [0, 1]^d, shape (n_trials, n_dims).
        center: Optimal center, shape (n_dims,).
        func_type: One of "quadratic", "rosenbrock", "ackley", "rastrigin".
        rng: Random state.
        n_dims: Number of dimensions.

    Returns:
        Objective values, shape (n_trials,).
    """
    # Scale from [0,1] to function-appropriate domains
    diff = configs - center[np.newaxis, :]

    if func_type == "quadratic":
        scales = rng.uniform(0.5, 5.0, size=n_dims)
        objectives = np.sum(scales * diff**2, axis=1)

    elif func_type == "rosenbrock":
        # Rosenbrock in shifted [0,1] space
        x = diff * 4.0  # scale to [-2, 2] range
        objectives = np.zeros(len(configs))
        for d in range(n_dims - 1):
            objectives += 100.0 * (x[:, d + 1] - x[:, d]**2)**2 + (1.0 - x[:, d])**2
        objectives /= n_dims  # normalize by dimension

    elif func_type == "ackley":
        x = diff * 4.0
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        sum_sq = np.mean(x**2, axis=1)
        sum_cos = np.mean(np.cos(c * x), axis=1)
        objectives = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e

    elif func_type == "rastrigin":
        x = diff * 5.12
        objectives = 10.0 * n_dims + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)
        objectives /= n_dims

    else:
        raise ValueError(f"Unknown function type: {func_type}")

    # Add small noise
    noise = rng.normal(0, 0.01 * (np.std(objectives) + 1e-6), size=len(objectives))
    return (objectives + noise).astype(np.float32)


def _extract_openml_meta_features(ds_id: str) -> Optional[np.ndarray]:
    """Extract Tier 1 + Tier 2 dataset meta-features from OpenML.

    HPO-B dataset IDs map to OpenML dataset IDs. This function fetches
    the dataset metadata and computes all META_FEATURE_DIM features.

    Tier 1 (slots 0-6): Basic dataset properties always available.
    Tier 2 (slots 7-9): Aggregate statistics causally linked to HPO
        landscape geometry. Gracefully defaults to 0.5 if unavailable.

    Args:
        ds_id: HPO-B dataset identifier (typically an OpenML dataset ID).

    Returns:
        Meta-features array of shape (META_FEATURE_DIM,), or None if
        extraction fails (network error, missing dataset, etc.).
    """
    try:
        import openml
        dataset = openml.datasets.get_dataset(int(ds_id), download_data=False)
        qualities = dataset.qualities

        # Extract raw values with safe defaults
        n_samples = float(qualities.get("NumberOfInstances", 1000))
        n_features = float(qualities.get("NumberOfFeatures", 10))
        n_classes = float(qualities.get("NumberOfClasses", 2))
        n_cat = float(qualities.get("NumberOfSymbolicFeatures", 0))
        n_missing = float(qualities.get("NumberOfMissingValues", 0))

        # Handle regression tasks (n_classes = 0 or NaN)
        if n_classes != n_classes or n_classes <= 0:  # NaN check
            n_classes = 0.0

        # ---- Tier 1: Basic dataset properties (slots 0-6) ----

        log_n_samples = np.log10(max(n_samples, 1.0)) / 6.0
        log_n_features = np.log10(max(n_features, 1.0)) / 4.0
        n_classes_norm = min(n_classes / 20.0, 1.0)
        dim_ratio = min(n_features / max(n_samples, 1.0), 1.0)

        # Imbalance ratio from qualities if available
        minority_size = float(qualities.get("MinorityClassSize", 0))
        majority_size = float(qualities.get("MajorityClassSize", 1))
        if majority_size > 0 and minority_size > 0:
            imbalance = minority_size / majority_size
        else:
            imbalance = 1.0  # balanced by default

        frac_cat = n_cat / max(n_features, 1.0)

        total_cells = n_samples * n_features
        frac_missing = n_missing / max(total_cells, 1.0)

        # ---- Tier 2: Aggregate statistics (slots 7-9) ----
        # These have direct causal links to HPO landscape geometry.
        # Default to 0.5 (uninformative prior) if unavailable.

        # Slot 7: Intrinsic dimensionality ratio
        # PCA effective dimensionality / n_features. Low values → data lives
        # on a low-dim manifold → regularization landscape is simpler.
        # OpenML stores the % variance explained by the 1st PC; we approximate
        # intrinsic dim as 1 / pct_1st_pc (capped, normalized).
        pct_1st_pc = qualities.get("PercentageOfFeatures1stPC", None)
        if pct_1st_pc is not None:
            pct_1st_pc = float(pct_1st_pc)
            if pct_1st_pc > 0:
                # Approximate: if 1st PC explains X%, effective dim ≈ 1/X
                # Normalize: intrinsic_dim / n_features
                approx_intrinsic_dim = min(1.0 / max(pct_1st_pc / 100.0, 0.01), n_features)
                intrinsic_dim_ratio = approx_intrinsic_dim / max(n_features, 1.0)
            else:
                intrinsic_dim_ratio = 0.5
        else:
            intrinsic_dim_ratio = 0.5  # uninformative default

        # Slot 8: Mean mutual information (features → target)
        # Captures signal-to-noise ratio. High MI → flatter HPO landscape.
        # OpenML stores MeanMutualInformation; normalize by target entropy.
        mean_mi = qualities.get("MeanMutualInformation", None)
        class_entropy = qualities.get("ClassEntropy", None)
        if mean_mi is not None and class_entropy is not None:
            mean_mi = float(mean_mi)
            class_entropy = float(class_entropy)
            if class_entropy > 0:
                norm_mi = min(mean_mi / class_entropy, 1.0)
            else:
                norm_mi = 0.5
        else:
            norm_mi = 0.5  # uninformative default

        # Slot 9: Landmark 1-NN accuracy
        # Dataset difficulty proxy. Easy datasets (high 1-NN acc) have flat
        # HPO landscapes; hard datasets have sharp, sensitive landscapes.
        landmark_1nn = qualities.get("kNN1NAcc", None)
        if landmark_1nn is None:
            landmark_1nn = qualities.get("Landmark1NN", None)
        if landmark_1nn is not None:
            landmark_1nn = float(landmark_1nn)
            # Already in [0, 1] for accuracy
        else:
            landmark_1nn = 0.5  # uninformative default

        return np.array([
            # Tier 1
            np.clip(log_n_samples, 0.0, 1.0),
            np.clip(log_n_features, 0.0, 1.0),
            np.clip(n_classes_norm, 0.0, 1.0),
            np.clip(dim_ratio, 0.0, 1.0),
            np.clip(imbalance, 0.0, 1.0),
            np.clip(frac_cat, 0.0, 1.0),
            np.clip(frac_missing, 0.0, 1.0),
            # Tier 2
            np.clip(intrinsic_dim_ratio, 0.0, 1.0),
            np.clip(norm_mi, 0.0, 1.0),
            np.clip(landmark_1nn, 0.0, 1.0),
        ], dtype=np.float32)

    except Exception as e:
        # Silently return None — caller will use zeros
        import logging
        logging.getLogger(__name__).debug(
            "Could not extract meta-features for ds_id=%s: %s", ds_id, e
        )
        return None


# Cache for OpenML meta-features to avoid repeated API calls
_meta_feature_cache: Dict[str, Optional[np.ndarray]] = {}


def _get_cached_meta_features(ds_id: str) -> Optional[np.ndarray]:
    """Get meta-features with caching to avoid duplicate OpenML API calls."""
    if ds_id not in _meta_feature_cache:
        _meta_feature_cache[ds_id] = _extract_openml_meta_features(ds_id)
    return _meta_feature_cache[ds_id]


def generate_hpob_trajectories(
    data_dir: str = "data/hpob",
    gamma: float = 0.15,
    seed: int = 42,
    include_meta_features: bool = True,
) -> List[TrajectoryPair]:
    """Generate training pairs from HPO-B surrogate benchmark.

    Requires hpob-handler to be installed. Falls back to synthetic
    data if HPO-B is not available.

    Args:
        data_dir: Path to HPO-B data directory.
        gamma: Quantile threshold.
        seed: Random seed.
        include_meta_features: If True, extract OpenML meta-features for
            each dataset (requires openml package, falls back gracefully).

    Returns:
        List of TrajectoryPair instances.
    """
    try:
        from hpob_handler import HPOBHandler
    except ImportError:
        print(
            "WARNING: hpob-handler not installed. "
            "Install with: pip install hpob-handler\n"
            "Falling back to synthetic data generation."
        )
        return generate_synthetic_trajectories(seed=seed, include_meta_features=include_meta_features)

    handler = HPOBHandler(root_dir=data_dir, mode="v3")
    preprocessor = TrajectoryPreprocessor(
        gamma=gamma,
        min_prefix_len=5,
        max_prefix_len=50,
        n_prefixes_per_trajectory=5,
        seed=seed,
    )

    all_pairs: List[TrajectoryPair] = []
    search_spaces = handler.get_search_spaces()

    for ss_id in tqdm(search_spaces, desc="Processing HPO-B search spaces"):
        datasets = handler.get_datasets(ss_id)
        for ds_id in datasets:
            try:
                # Get pre-evaluated configurations and objectives
                if hasattr(handler, 'meta_train_data') and ss_id in handler.meta_train_data and ds_id in handler.meta_train_data[ss_id]:
                    data_dict = handler.meta_train_data[ss_id][ds_id]
                elif hasattr(handler, 'meta_test_data') and ss_id in handler.meta_test_data and ds_id in handler.meta_test_data[ss_id]:
                    data_dict = handler.meta_test_data[ss_id][ds_id]
                elif hasattr(handler, 'bo_initializations') and ss_id in handler.bo_initializations and ds_id in handler.bo_initializations[ss_id]:
                    data_dict = handler.bo_initializations[ss_id][ds_id]
                else:
                    raise ValueError(f"Data missing for {ss_id}/{ds_id}")
                
                X, y = data_dict["X"], data_dict["y"]
                configs = np.array(X, dtype=np.float32)
                objectives = np.array(y, dtype=np.float32).flatten()

                # Normalize configs to [0, 1] per dimension
                c_min = configs.min(axis=0)
                c_max = configs.max(axis=0)
                valid = (c_max - c_min) > 1e-8
                configs[:, valid] = (configs[:, valid] - c_min[valid]) / (c_max[valid] - c_min[valid])
                configs[:, ~valid] = 0.5

                # Extract dataset-level meta-features from OpenML
                meta = None
                if include_meta_features:
                    meta = _get_cached_meta_features(ds_id)

                pairs = preprocessor.process_trajectory(
                    configs=configs,
                    objectives=objectives,
                    search_space_id=f"hpob_{ss_id}_{ds_id}",
                    minimize=True,
                    meta_features=meta,
                )
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"  Skipping {ss_id}/{ds_id}: {e}")
                continue

    return all_pairs


def save_pairs(pairs: List[TrajectoryPair], output_path: str) -> None:
    """Save trajectory pairs to disk as a serialized file.

    Args:
        pairs: List of TrajectoryPair instances.
        output_path: Output file path (.pt format).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = []
    for p in pairs:
        item = {
            "input_configs": p.input_configs,
            "input_objectives": p.input_objectives,
            "target_lower": p.target_lower,
            "target_upper": p.target_upper,
            "search_space_id": p.search_space_id,
            "seq_len": p.seq_len,
        }
        if p.meta_features is not None:
            item["meta_features"] = p.meta_features
        serialized.append(item)

    torch.save(serialized, output_path)
    print(f"Saved {len(serialized)} training pairs to {output_path}")


def load_pairs(input_path: str) -> List[TrajectoryPair]:
    """Load trajectory pairs from disk.

    Args:
        input_path: Path to .pt file.

    Returns:
        List of TrajectoryPair instances.
    """
    data = torch.load(input_path, weights_only=False)
    pairs = []
    for item in data:
        pairs.append(TrajectoryPair(
            input_configs=item["input_configs"],
            input_objectives=item["input_objectives"],
            target_lower=item["target_lower"],
            target_upper=item["target_upper"],
            search_space_id=item["search_space_id"],
            seq_len=item["seq_len"],
            meta_features=item.get("meta_features", None),
        ))
    return pairs


def main():
    """CLI entry point for trajectory generation."""
    parser = argparse.ArgumentParser(
        description="Generate HPO trajectory training data for TRIP-TPE"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "hpob"],
        default="synthetic",
        help="Data source: synthetic functions or HPO-B benchmark",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=50000,
        help="Number of trajectories to generate (synthetic mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train.pt",
        help="Output file path",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.15,
        help="Quantile threshold for good region",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--hpob-dir",
        type=str,
        default="data/hpob",
        help="HPO-B data directory (hpob mode)",
    )
    parser.add_argument("--wandb-project", type=str, default="trip-tpe", help="W&B project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Initialize W&B for data generation tracking
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity else None,
                job_type="data-generation",
                config={
                    "mode": args.mode,
                    "n_trajectories": args.n_trajectories,
                    "gamma": args.gamma,
                    "seed": args.seed,
                    "output": args.output,
                },
                tags=["data-gen", args.mode],
            )
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging.")
            use_wandb = False

    import time
    t0 = time.time()

    if args.mode == "synthetic":
        pairs = generate_synthetic_trajectories(
            n_trajectories=args.n_trajectories,
            gamma=args.gamma,
            seed=args.seed,
        )
    elif args.mode == "hpob":
        pairs = generate_hpob_trajectories(
            data_dir=args.hpob_dir,
            gamma=args.gamma,
            seed=args.seed,
        )

    save_pairs(pairs, args.output)

    gen_time = time.time() - t0

    # Log data generation summary to W&B
    if use_wandb:
        # Compute dataset statistics
        seq_lens = [p.seq_len for p in pairs]
        n_dims_list = [p.input_configs.shape[1] for p in pairs]
        has_meta = sum(1 for p in pairs if p.meta_features is not None)
        wandb.log({
            "data/n_pairs": len(pairs),
            "data/mean_seq_len": float(np.mean(seq_lens)),
            "data/max_seq_len": int(np.max(seq_lens)),
            "data/mean_n_dims": float(np.mean(n_dims_list)),
            "data/pairs_with_meta_features": has_meta,
            "data/generation_time_s": gen_time,
        })
        # Register output as artifact
        artifact = wandb.Artifact(
            f"trip-tpe-data-{args.mode}", type="dataset",
            description=f"{len(pairs)} trajectory pairs from {args.mode} source",
        )
        artifact.add_file(args.output)
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
