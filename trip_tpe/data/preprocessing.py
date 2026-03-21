"""
Trajectory preprocessing for TRIP-TPE.

Converts raw HPO evaluation histories into (partial_trajectory → top-performing_region)
supervised training pairs. This is the core data transformation that enables the
Transformer to learn coarse region proposals from optimization history patterns.

Horizon-aware labelling (v0.1.1):
    Instead of computing a single target region from the FULL trajectory for all
    prefixes, each prefix now receives a label computed from trials[0 : plen + lookahead],
    where lookahead = lookahead_multiplier * plen (capped at the trajectory length).
    This ensures:
        - Short prefixes receive broad, near-horizon targets the model can plausibly
          infer from early trial patterns (reduces gradient noise).
        - Long prefixes receive tighter, far-horizon targets that converge toward the
          full-trajectory region (trains the model to refine proposals).
        - The model learns *progressive refinement* as a function of input length,
          matching the requery-based inference loop in the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Number of dataset-level meta-features used for cold-start conditioning.
#
# Tier 1 (always computable from basic dataset metadata):
#   0: log10(n_samples) / 6    — dataset size (log-scaled, ~[0,1])
#   1: log10(n_features) / 4   — feature count (log-scaled, ~[0,1])
#   2: n_classes / 20          — classification cardinality (capped, /20)
#   3: n_features / n_samples  — dimensionality ratio (capped at 1.0)
#   4: imbalance_ratio         — minority_class_frac / majority_class_frac (0-1)
#   5: frac_categorical        — fra#ction of categorical features (0-1)
#   6: frac_missing            — fraction of missing values (0-1)
#
# Tier 2 (aggregate statistics causally linked to HPO landscape geometry):
#   7: intrinsic_dim_ratio     — PCA effective dim / n_features (0-1).
#                                 Captures whether the feature space is truly
#                                 high-dimensional or lives on a low-dim manifold.
#                                 Directly governs regularization landscape shape.
#   8: mean_mutual_info        — avg MI(feature, target) / H(target) (0-1).
#                                 Captures signal-to-noise ratio. High MI → flatter
#                                 HPO landscape (many configs work). Low MI → sharp
#                                 ridges, narrow optima.
#   9: landmark_1nn_accuracy   — 1-NN accuracy on the dataset (0-1).
#                                 Dataset difficulty proxy. Easy datasets have flat
#                                 HPO landscapes; hard datasets have sensitive ones.
META_FEATURE_DIM = 10


@dataclass
class TrajectoryPair:
    """A single training example for the region proposal model.

    Attributes:
        input_configs: Normalized hyperparameter configs, shape (seq_len, n_dims).
        input_objectives: Objective values, shape (seq_len, 1).
        target_lower: Lower bounds of top-performing region, shape (n_dims,).
        target_upper: Upper bounds of top-performing region, shape (n_dims,).
        search_space_id: Identifier for the search space (for stratified sampling).
        seq_len: Actual sequence length (before padding).
        meta_features: Optional dataset-level meta-features, shape (META_FEATURE_DIM,).
                       None means no meta-features available (model uses zeros).
    """
    input_configs: np.ndarray
    input_objectives: np.ndarray
    target_lower: np.ndarray
    target_upper: np.ndarray
    search_space_id: str
    seq_len: int
    meta_features: Optional[np.ndarray] = None


class TrajectoryPreprocessor:
    """Preprocesses raw HPO trajectories into supervised training pairs.

    The core preprocessing pipeline:
    1. Normalize all hyperparameter configurations to [0, 1] per dimension.
    2. Normalize objective values via rank normalization (robust to outliers).
    3. Extract the top-γ quantile of configurations as the "good" region.
    4. Compute axis-aligned bounding box of the good region → target bounds.
    5. Sample partial prefixes of the trajectory as input sequences.
    """

    def __init__(
        self,
        gamma: float = 0.15,
        min_prefix_len: int = 5,
        max_prefix_len: int = 50,
        n_prefixes_per_trajectory: int = 5,
        region_padding: float = 0.05,
        seed: int = 42,
        lookahead_multiplier: float = 2.0,
        use_horizon_aware_labels: bool = True,
    ):
        """Initialize the preprocessor.

        Args:
            gamma: Quantile threshold for defining "good" configurations.
                   Aligned with Watanabe 2023 finding that γ=0.10-0.20 is optimal.
            min_prefix_len: Minimum number of trials in a prefix.
            max_prefix_len: Maximum prefix length.
            n_prefixes_per_trajectory: Number of prefix samples per trajectory.
            region_padding: Fractional padding added to bounding boxes.
            seed: Random seed for reproducibility.
            lookahead_multiplier: For horizon-aware labels, the target region for
                a prefix of length P is computed from trials[0 : P + P*multiplier].
                Default 2.0 means the model sees P trials and the label covers 3P
                trials total. Higher values → more forward-looking labels.
            use_horizon_aware_labels: If True (default), each prefix gets a
                target region computed from its own horizon window. If False,
                falls back to the original behaviour (all prefixes share the
                full-trajectory label).
        """
        self.gamma = gamma
        self.min_prefix_len = min_prefix_len
        self.max_prefix_len = max_prefix_len
        self.n_prefixes_per_trajectory = n_prefixes_per_trajectory
        self.region_padding = region_padding
        self.rng = np.random.RandomState(seed)
        self.lookahead_multiplier = lookahead_multiplier
        self.use_horizon_aware_labels = use_horizon_aware_labels

    def normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Rank-normalize objective values to [0, 1].

        Rank normalization is robust to outliers and heterogeneous
        objective scales across different HPO tasks.

        Args:
            objectives: Raw objective values, shape (n_trials,).

        Returns:
            Rank-normalized values in [0, 1], shape (n_trials,).
        """
        n = len(objectives)
        if n <= 1:
            return np.zeros_like(objectives)
        ranks = np.argsort(np.argsort(objectives)).astype(np.float32)
        return ranks / (n - 1)

    def compute_target_region(
        self,
        configs: np.ndarray,
        objectives: np.ndarray,
        minimize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the axis-aligned bounding box of top-γ configurations.

        This defines the "ground truth" region that the Transformer should predict.

        Args:
            configs: Normalized configs, shape (n_trials, n_dims).
            objectives: Raw objective values, shape (n_trials,).
            minimize: If True, lower objective values are better.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (n_dims,).
        """
        n_trials = len(objectives)
        n_good = max(1, int(n_trials * self.gamma))

        if minimize:
            good_indices = np.argsort(objectives)[:n_good]
        else:
            good_indices = np.argsort(objectives)[-n_good:]

        good_configs = configs[good_indices]

        # Axis-aligned bounding box
        lower = np.min(good_configs, axis=0)
        upper = np.max(good_configs, axis=0)

        # Add padding to prevent overly tight bounds
        width = upper - lower
        padding = np.maximum(width * self.region_padding, self.region_padding)
        lower = np.clip(lower - padding, 0.0, 1.0)
        upper = np.clip(upper + padding, 0.0, 1.0)

        # Ensure minimum region width
        min_width = 0.05
        too_narrow = (upper - lower) < min_width
        midpoints = (upper + lower) / 2.0
        lower[too_narrow] = np.clip(midpoints[too_narrow] - min_width / 2, 0.0, 1.0)
        upper[too_narrow] = np.clip(midpoints[too_narrow] + min_width / 2, 0.0, 1.0)

        return lower, upper

    def _compute_horizon(self, prefix_len: int, n_trials: int) -> int:
        """Compute the horizon endpoint for a given prefix length.

        The horizon defines how many trials (from the start of the trajectory)
        contribute to the target region for this prefix.

        For a prefix of length P, the horizon is:
            min(P + floor(P * lookahead_multiplier), n_trials)

        With the default multiplier of 2.0, a prefix of 10 gets a horizon of
        30 trials, and a prefix of 40 gets a horizon of 120 (capped at
        n_trials). This means longer prefixes naturally get labels closer to
        the full-trajectory region.

        Args:
            prefix_len: Number of trials the model will see as input.
            n_trials: Total trials in the trajectory.

        Returns:
            Horizon endpoint (inclusive), i.e. target uses trials[0:horizon].
        """
        lookahead = int(prefix_len * self.lookahead_multiplier)
        return min(prefix_len + lookahead, n_trials)

    def process_trajectory(
        self,
        configs: np.ndarray,
        objectives: np.ndarray,
        search_space_id: str = "unknown",
        minimize: bool = True,
        meta_features: Optional[np.ndarray] = None,
    ) -> List[TrajectoryPair]:
        """Process a single complete HPO trajectory into training pairs.

        When horizon-aware labels are enabled (default), each prefix receives
        a target region computed from trials[0 : prefix_len + lookahead],
        where lookahead = prefix_len * lookahead_multiplier. This teaches the
        model to make progressively more precise region predictions as it
        receives more input data — matching the requery pattern in the sampler.

        When disabled, all prefixes share a single target region computed from
        the full trajectory (original v0.1.0 behaviour).

        Args:
            configs: Normalized configs, shape (n_trials, n_dims).
            objectives: Raw objective values, shape (n_trials,).
            search_space_id: Identifier for this search space.
            minimize: If True, lower objective values are better.
            meta_features: Optional dataset-level meta-features, shape
                          (META_FEATURE_DIM,). Propagated to all TrajectoryPair
                          instances generated from this trajectory.

        Returns:
            List of TrajectoryPair instances.
        """
        n_trials, n_dims = configs.shape
        if n_trials < self.min_prefix_len + 5:
            return []

        # Normalize objectives for input (using rank normalization)
        norm_objectives = self.normalize_objectives(objectives)

        # Pre-compute the full-trajectory region (used when horizon-aware
        # labels are disabled, and as the ceiling for long prefixes).
        full_target_lower, full_target_upper = self.compute_target_region(
            configs, objectives, minimize
        )

        # Sample prefix lengths
        max_len = min(self.max_prefix_len, n_trials - 1)
        if max_len <= self.min_prefix_len:
            prefix_lengths = [self.min_prefix_len]
        else:
            prefix_lengths = self.rng.randint(
                self.min_prefix_len,
                max_len + 1,
                size=self.n_prefixes_per_trajectory,
            )

        pairs = []
        for plen in prefix_lengths:
            plen = int(plen)
            if plen > n_trials:
                plen = n_trials

            # Compute the target region for this specific prefix
            if self.use_horizon_aware_labels:
                horizon = self._compute_horizon(plen, n_trials)
                # Need enough trials in the horizon window for a meaningful
                # target. If horizon is too short, fall back to full trajectory.
                if horizon >= plen + 2:
                    target_lower, target_upper = self.compute_target_region(
                        configs[:horizon], objectives[:horizon], minimize
                    )
                else:
                    target_lower = full_target_lower
                    target_upper = full_target_upper
            else:
                target_lower = full_target_lower
                target_upper = full_target_upper

            pairs.append(TrajectoryPair(
                input_configs=configs[:plen].astype(np.float32),
                input_objectives=norm_objectives[:plen].reshape(-1, 1).astype(np.float32),
                target_lower=target_lower.astype(np.float32),
                target_upper=target_upper.astype(np.float32),
                search_space_id=search_space_id,
                seq_len=plen,
                meta_features=meta_features,
            ))

        return pairs

    def process_batch(
        self,
        trajectories: List[Dict],
        minimize: bool = True,
    ) -> List[TrajectoryPair]:
        """Process a batch of trajectories.

        Args:
            trajectories: List of dicts with keys:
                - "configs": np.ndarray of shape (n_trials, n_dims)
                - "objectives": np.ndarray of shape (n_trials,)
                - "search_space_id": str (optional)
                - "meta_features": np.ndarray (optional), shape (META_FEATURE_DIM,)
            minimize: If True, lower is better.

        Returns:
            Flat list of all TrajectoryPair instances.
        """
        all_pairs = []
        for traj in trajectories:
            pairs = self.process_trajectory(
                configs=traj["configs"],
                objectives=traj["objectives"],
                search_space_id=traj.get("search_space_id", "unknown"),
                minimize=minimize,
                meta_features=traj.get("meta_features", None),
            )
            all_pairs.extend(pairs)
        return all_pairs
