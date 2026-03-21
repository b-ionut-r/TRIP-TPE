"""
TRIP-TPE Sampler: The core Optuna sampler integrating Transformer region proposals with TPE.

This is the primary user-facing component — a drop-in replacement for Optuna's TPESampler
that augments standard TPE with learned region proposals from a pre-trained Transformer.

Operational Flow:
    1. For the first N trials (n_warmup_trials), sample uniformly at random.
    2. After warmup, query the Transformer with the observed trajectory to obtain
       per-dimension region bounds.
    3. Construct a constrained search space from the proposed region.
    4. Delegate sampling within the constrained space to Optuna's TPESampler.
    5. Periodically re-query the Transformer (every requery_interval trials) to
       update the region as more data accumulates.

Key Design Decisions:
    - The Transformer is ONLY used for macro-region definition (coarse guidance).
    - TPE handles ALL fine-grained density estimation and sampling.
    - This separation preserves TPE's native support for categorical/conditional spaces
      while adding the Transformer's cross-task generalization capability.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TRIPTPESampler(BaseSampler):
    """Trajectory-Informed Region Proposal + TPE Sampler.

    A hybrid sampler that uses a pre-trained Transformer to propose promising
    search space regions, then delegates fine-grained sampling to TPE within
    those constrained regions.

    Example:
        >>> import optuna
        >>> from trip_tpe import TRIPTPESampler
        >>>
        >>> sampler = TRIPTPESampler(
        ...     model_path="checkpoints/trip_tpe_best.pt",
        ...     n_warmup_trials=5,
        ... )
        >>> study = optuna.create_study(sampler=sampler)
        >>> study.optimize(objective, n_trials=100)

    Without a pre-trained model (falls back to pure TPE):
        >>> sampler = TRIPTPESampler()  # No model_path → graceful degradation
        >>> study = optuna.create_study(sampler=sampler)
        >>> study.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        # Region proposal settings
        n_warmup_trials: int = 5,
        requery_interval: int = 10,
        confidence_level: float = 0.9,
        min_region_fraction: float = 0.05,
        max_region_fraction: float = 0.8,
        trust_factor: float = 0.8,
        trust_decay: float = 0.995,
        min_trust_factor: float = 0.3,
        # TPE settings
        n_startup_trials: int = 0,
        n_ei_candidates: int = 24,
        gamma: Optional[float] = None,
        multivariate: bool = True,
        group: bool = True,
        constant_liar: bool = False,
        # General
        seed: Optional[int] = None,
        device: str = "auto",
        hp_dim: int = 16,
        max_seq_len: int = 200,
        beta_blend_weight: float = 0.5,
        # P0: Adaptive trust scheduling
        adaptive_trust: bool = False,
        adaptive_trust_target_coverage: float = 0.8,
        adaptive_trust_sensitivity: float = 5.0,
        adaptive_trust_gamma: float = 0.15,
        # Meta-features for cold-start conditioning
        meta_features: Optional[np.ndarray] = None,
    ):
        """Initialize the TRIP-TPE sampler.

        Args:
            model_path: Path to pre-trained Transformer checkpoint (.pt file).
                        If None, operates as enhanced TPE without region proposals.
            n_warmup_trials: Number of random trials before Transformer activation.
            requery_interval: Re-query Transformer every N trials.
            confidence_level: Confidence level for region bound quantiles.
            min_region_fraction: Minimum region width per dimension.
            max_region_fraction: Maximum region width per dimension.
            trust_factor: Initial trust in Transformer proposals (0-1).
            trust_decay: Multiplicative decay for trust factor per requery.
            min_trust_factor: Floor for trust factor.
            n_startup_trials: TPE startup trials (0 since Transformer handles warmup).
            n_ei_candidates: Number of EI candidates for TPE.
            gamma: TPE gamma quantile. None = Optuna default.
            multivariate: Use multivariate TPE.
            group: Use group decomposition in TPE.
            constant_liar: Use constant liar for parallel sampling.
            seed: Random seed.
            device: Device for Transformer inference ("auto", "cuda", "cpu").
            hp_dim: Maximum hyperparameter dimensions the model supports.
            max_seq_len: Maximum trajectory sequence length for Transformer input.
            beta_blend_weight: Weight for Beta quantile bounds in the ensemble
                               (1.0 = only Beta, 0.0 = only direct bounds).
            meta_features: Optional dataset-level meta-features for cold-start
                          conditioning. Shape (META_FEATURE_DIM,) = (10,).
                          Tier 1 (slots 0-6): log10(n_samples)/6, log10(n_features)/4,
                          n_classes/20, dimensionality_ratio, imbalance_ratio,
                          frac_categorical, frac_missing.
                          Tier 2 (slots 7-9): intrinsic_dim_ratio, mean_mutual_info,
                          landmark_1nn_accuracy.
                          All values in [0, 1]. When provided, the Transformer uses
                          these to make better region proposals before observing many
                          trials (cold-start).
        """
        self._model_path = model_path
        self._n_warmup_trials = n_warmup_trials
        self._requery_interval = requery_interval
        self._confidence_level = confidence_level
        self._min_region_fraction = min_region_fraction
        self._max_region_fraction = max_region_fraction
        self._trust_factor_init = trust_factor
        self._trust_decay = trust_decay
        self._min_trust_factor = min_trust_factor
        self._hp_dim = hp_dim
        self._max_seq_len = max_seq_len
        self._beta_blend_weight = beta_blend_weight
        self._seed = seed

        # P0: Adaptive trust scheduling
        self._adaptive_trust = adaptive_trust
        self._adaptive_trust_target_coverage = adaptive_trust_target_coverage
        self._adaptive_trust_sensitivity = adaptive_trust_sensitivity
        self._adaptive_trust_gamma = adaptive_trust_gamma

        # Meta-features for cold-start conditioning
        self._meta_features = meta_features

        # Resolve device
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

        # Initialize the underlying TPE sampler
        tpe_kwargs: Dict[str, Any] = {
            "n_startup_trials": n_startup_trials,
            "n_ei_candidates": n_ei_candidates,
            "multivariate": multivariate,
            "group": group,
            "constant_liar": constant_liar,
            "seed": seed,
        }
        # Only pass gamma if explicitly set (Optuna handles default internally)
        # Use consider_endpoints for better boundary handling
        self._tpe_sampler = TPESampler(**tpe_kwargs)

        # State
        self._model: Optional[Any] = None  # Lazy-loaded Transformer
        self._model_loaded = False
        self._current_trust_factor = trust_factor
        self._last_requery_trial = 0
        self._cached_lower: Optional[np.ndarray] = None
        self._cached_upper: Optional[np.ndarray] = None
        self._encoder: Optional[Any] = None  # SearchSpaceEncoder, built on first call
        self._search_space_built = False
        self._n_requery_count = 0

    def _load_model(self) -> bool:
        """Lazy-load the pre-trained Transformer model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model_loaded:
            return self._model is not None

        if self._model_path is None:
            self._model_loaded = True
            return False

        if not TORCH_AVAILABLE:
            warnings.warn(
                "PyTorch not available. TRIP-TPE will fall back to standard TPE.",
                RuntimeWarning,
            )
            self._model_loaded = True
            return False

        try:
            from trip_tpe.models.region_proposal_transformer import RegionProposalTransformer

            checkpoint = torch.load(
                self._model_path,
                map_location=self._device,
                weights_only=False,
            )

            # Handle both raw state_dict and full checkpoint formats
            if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                config = checkpoint["model_config"]
                self._model = RegionProposalTransformer(**config)
                self._model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model = RegionProposalTransformer(hp_dim=self._hp_dim)
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Assume it's a raw state dict
                self._model = RegionProposalTransformer(hp_dim=self._hp_dim)
                self._model.load_state_dict(checkpoint)

            self._model.to(self._device)
            self._model.eval()
            self._model_loaded = True
            return True

        except Exception as e:
            warnings.warn(
                f"Failed to load TRIP-TPE model from {self._model_path}: {e}. "
                "Falling back to standard TPE.",
                RuntimeWarning,
            )
            self._model = None
            self._model_loaded = True
            return False

    def _build_encoder(
        self, search_space: Dict[str, BaseDistribution]
    ) -> None:
        """Build the search space encoder on first invocation.

        Args:
            search_space: Optuna search space dictionary.
        """
        if self._search_space_built:
            return

        from trip_tpe.utils.search_space import SearchSpaceEncoder
        self._encoder = SearchSpaceEncoder(search_space)
        self._search_space_built = True

    def _encode_trajectory(
        self, study: Study, search_space: Dict[str, BaseDistribution]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Encode the current optimization history as a Transformer-ready sequence.

        Args:
            study: Optuna study with completed trials.
            search_space: Current search space.

        Returns:
            Tuple of (input_seq, attention_mask, actual_len).
            input_seq: shape (1, max_seq_len, hp_dim + 1)
            attention_mask: shape (1, max_seq_len)
        """
        assert self._encoder is not None

        trials = study.trials

        # Separate completed from non-completed for logging
        completed = []
        n_pruned = 0
        n_failed = 0
        for t in trials:
            if t.state == TrialState.COMPLETE:
                completed.append(t)
            elif t.state == TrialState.PRUNED:
                n_pruned += 1
            elif t.state == TrialState.FAIL:
                n_failed += 1

        if n_pruned > 0 or n_failed > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "TRIP-TPE trajectory encoding: skipping %d pruned and %d failed "
                "trials (using %d completed trials).",
                n_pruned, n_failed, len(completed),
            )

        if not completed:
            return (
                np.zeros((1, 1, self._hp_dim + 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                0,
            )

        # Sort by trial number
        completed = sorted(completed, key=lambda t: t.number)

        # Encode each trial
        configs = []
        objectives = []
        _warned_multi_obj = False
        for trial in completed:
            encoded = self._encoder.encode_params(trial.params)
            # Pad to hp_dim
            padded = np.zeros(self._hp_dim, dtype=np.float32)
            n = min(len(encoded), self._hp_dim)
            padded[:n] = encoded[:n]
            configs.append(padded)

            # Use first objective value (single-objective only).
            # Guard against multi-objective studies and missing values.
            if trial.values is None or len(trial.values) == 0:
                objectives.append(0.0)
            else:
                if len(trial.values) > 1 and not _warned_multi_obj:
                    warnings.warn(
                        "TRIP-TPE currently supports single-objective optimization only. "
                        f"Using the first of {len(trial.values)} objectives.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _warned_multi_obj = True
                objectives.append(float(trial.values[0]))

        configs_arr = np.array(configs, dtype=np.float32)
        obj_arr = np.array(objectives, dtype=np.float32)

        # Rank-normalize objectives, respecting optimization direction.
        # For minimization, lower values should get lower ranks (closer to 0).
        # For maximization, higher values should get lower ranks (closer to 0).
        # This ensures the Transformer always sees 0 = "good" and 1 = "bad".
        n = len(obj_arr)
        if n > 1:
            # Determine direction — default to minimize for single-objective
            direction = getattr(study, "direction", None)
            if direction is not None and direction == optuna.study.StudyDirection.MAXIMIZE:
                # Flip: highest value → rank 0 (best)
                ranks = np.argsort(np.argsort(-obj_arr)).astype(np.float32)
            else:
                # Default: lowest value → rank 0 (best)
                ranks = np.argsort(np.argsort(obj_arr)).astype(np.float32)
            norm_obj = (ranks / (n - 1)).reshape(-1, 1)
        else:
            norm_obj = np.array([[0.5]], dtype=np.float32)

        # Concatenate to form input sequence
        seq = np.concatenate([configs_arr, norm_obj], axis=1)  # (n, hp_dim+1)

        # Truncate to max_seq_len (use model's configured value, not hardcoded)
        max_len = self._max_seq_len
        actual_len = min(n, max_len)
        seq = seq[:actual_len]

        # Pad to fixed length
        padded_seq = np.zeros((1, max_len, self._hp_dim + 1), dtype=np.float32)
        padded_seq[0, :actual_len] = seq
        mask = np.zeros((1, max_len), dtype=np.float32)
        mask[0, :actual_len] = 1.0

        return padded_seq, mask, actual_len

    def _query_transformer(
        self,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the Transformer for region proposals.

        Args:
            study: Current Optuna study.
            search_space: Current search space.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (n_dims,).
            Values in [0, 1] normalized space.
        """
        assert self._model is not None and self._encoder is not None

        input_seq, mask, actual_len = self._encode_trajectory(study, search_space)

        if actual_len == 0:
            n_dims = self._encoder.n_dims
            return np.zeros(n_dims, dtype=np.float32), np.ones(n_dims, dtype=np.float32)

        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_seq).to(self._device)
        mask_tensor = torch.from_numpy(mask).to(self._device)

        # Dimension mask
        n_dims = self._encoder.n_dims
        dim_mask = np.zeros((1, self._hp_dim), dtype=np.float32)
        dim_mask[0, :min(n_dims, self._hp_dim)] = 1.0
        dim_mask_tensor = torch.from_numpy(dim_mask).to(self._device)

        # Prepare meta-features tensor if available
        meta_tensor = None
        if self._meta_features is not None:
            meta_np = np.array(self._meta_features, dtype=np.float32).reshape(1, -1)
            meta_tensor = torch.from_numpy(meta_np).to(self._device)

        # Get predictions
        lower, upper = self._model.predict_region(
            input_tensor,
            attention_mask=mask_tensor,
            confidence_level=self._confidence_level,
            dim_mask=dim_mask_tensor,
            beta_blend_weight=self._beta_blend_weight,
            meta_features=meta_tensor,
        )

        lower_np = lower[0, :n_dims].cpu().numpy()
        upper_np = upper[0, :n_dims].cpu().numpy()

        # Apply trust factor blending with full search space
        tf = self._current_trust_factor
        lower_np = tf * lower_np + (1 - tf) * 0.0
        upper_np = tf * upper_np + (1 - tf) * 1.0

        # Enforce min/max region fractions
        width = upper_np - lower_np
        too_narrow = width < self._min_region_fraction
        too_wide = width > self._max_region_fraction

        midpoints = (lower_np + upper_np) / 2.0

        # Expand narrow regions
        lower_np[too_narrow] = np.clip(
            midpoints[too_narrow] - self._min_region_fraction / 2, 0.0, 1.0
        )
        upper_np[too_narrow] = np.clip(
            midpoints[too_narrow] + self._min_region_fraction / 2, 0.0, 1.0
        )

        # Shrink overly wide regions
        lower_np[too_wide] = np.clip(
            midpoints[too_wide] - self._max_region_fraction / 2, 0.0, 1.0
        )
        upper_np[too_wide] = np.clip(
            midpoints[too_wide] + self._max_region_fraction / 2, 0.0, 1.0
        )

        return lower_np, upper_np

    def _compute_adaptive_trust(
        self, study: Study
    ) -> float:
        """P0: Compute coverage-aware trust factor adjustment.

        Monitors the fraction of top-gamma trials that fall within the
        Transformer's proposed region. If coverage is high, trust stays
        strong. If coverage is low, trust decays faster.

        Args:
            study: Current Optuna study.

        Returns:
            Adjusted trust factor.
        """
        if self._cached_lower is None or self._cached_upper is None:
            return self._current_trust_factor
        if self._encoder is None:
            return self._current_trust_factor

        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        if len(completed) < 3:
            return self._current_trust_factor

        # Determine the top-gamma trials, respecting optimization direction
        gamma = self._adaptive_trust_gamma
        n_good = max(1, int(len(completed) * gamma))
        values = [t.value for t in completed if t.value is not None]
        if not values:
            return self._current_trust_factor

        # For minimization: lowest values are best → sort ascending, take first n.
        # For maximization: highest values are best → sort descending, take first n.
        direction = getattr(study, "direction", None)
        descending = (
            direction is not None
            and direction == optuna.study.StudyDirection.MAXIMIZE
        )
        sorted_trials = sorted(
            [t for t in completed if t.value is not None],
            key=lambda t: t.value,
            reverse=descending,
        )
        good_trials = sorted_trials[:n_good]

        # Check how many good trials fall within the proposed region
        n_inside = 0
        for trial in good_trials:
            encoded = self._encoder.encode_params(trial.params)
            n_dims = min(len(encoded), len(self._cached_lower))
            inside = True
            for d in range(n_dims):
                if encoded[d] < self._cached_lower[d] or encoded[d] > self._cached_upper[d]:
                    inside = False
                    break
            if inside:
                n_inside += 1

        coverage_rate = n_inside / max(len(good_trials), 1)

        # Sigmoid-based trust adjustment
        target = self._adaptive_trust_target_coverage
        k = self._adaptive_trust_sensitivity
        # When coverage_rate > target → adjustment > 0.5 → trust sustained
        # When coverage_rate < target → adjustment < 0.5 → trust decays
        import math
        adjustment = 1.0 / (1.0 + math.exp(-k * (coverage_rate - target)))

        # Scale: adjustment=0.5 → no change; >0.5 → slower decay; <0.5 → faster decay
        decay_multiplier = 0.5 + adjustment  # Range [0.5, 1.5]
        new_trust = self._current_trust_factor * (self._trust_decay ** decay_multiplier)
        new_trust = max(self._min_trust_factor, new_trust)

        return new_trust

    def _apply_joint_hypervolume_guard(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Guard against joint hypervolume collapse.

        If the product of per-dimension widths falls below a dimension-aware
        threshold, proportionally expand all dimensions to restore minimum volume.

        The threshold scales with dimensionality to prevent the guard from
        over-expanding in high-dimensional spaces (where even moderate per-dim
        widths produce tiny joint volumes).

        Args:
            lower: Lower bounds, shape (n_dims,).
            upper: Upper bounds, shape (n_dims,).

        Returns:
            Adjusted (lower, upper) bounds.
        """
        widths = upper - lower
        # Avoid log(0) — clamp widths to small positive value
        safe_widths = np.maximum(widths, 1e-12)
        log_volume = np.sum(np.log(safe_widths))

        # Dimension-aware threshold: min_region_fraction^n_dims gives the volume
        # when every dimension is at its minimum allowed width. The guard should
        # only trigger below this natural floor, not above it.
        #
        # NOTE: We use ONLY the dimension-aware threshold here. A fixed scalar
        # (e.g. 1e-6) dominates at d>12 and causes systematic over-expansion
        # that negates the Transformer's proposals in high-dimensional spaces.
        n_dims = len(widths)
        dimension_aware_threshold = self._min_region_fraction ** n_dims
        log_threshold = np.log(max(dimension_aware_threshold, 1e-300))

        if log_volume >= log_threshold:
            return lower, upper  # Volume is fine

        # Volume is too small — expand proportionally
        n_dims = len(widths)
        if n_dims == 0:
            return lower, upper

        # Compute required per-dimension expansion factor
        # We want: product(widths * factor) >= threshold
        # factor^n * product(widths) >= threshold
        # factor >= (threshold / product(widths))^(1/n)
        log_deficit = (log_threshold - log_volume) / n_dims
        expansion_factor = np.exp(log_deficit)

        # Apply expansion centered on midpoints
        midpoints = (lower + upper) / 2.0
        new_half_widths = safe_widths * expansion_factor / 2.0
        lower_expanded = np.clip(midpoints - new_half_widths, 0.0, 1.0)
        upper_expanded = np.clip(midpoints + new_half_widths, 0.0, 1.0)

        return lower_expanded, upper_expanded

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        """Infer the search space for relative sampling.

        Delegates to TPE's search space inference.
        """
        return self._tpe_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Sample parameters using Transformer-guided TPE.

        This is the main sampling method called by Optuna.

        Args:
            study: Current study.
            trial: Current trial being sampled.
            search_space: Search space to sample from.

        Returns:
            Dictionary of sampled parameter values.
        """
        if not search_space:
            return {}

        n_completed = len([
            t for t in study.trials if t.state == TrialState.COMPLETE
        ])

        # Phase 1: Warmup — use random sampling
        if n_completed < self._n_warmup_trials:
            return self._tpe_sampler.sample_relative(study, trial, search_space)

        # Build encoder if needed
        self._build_encoder(search_space)

        # Load model if needed
        has_model = self._load_model()

        if not has_model:
            # No model available — fall back to standard TPE
            return self._tpe_sampler.sample_relative(study, trial, search_space)

        # Phase 2: Query Transformer for region proposals (or use cached)
        should_requery = (
            self._cached_lower is None
            or (n_completed - self._last_requery_trial) >= self._requery_interval
        )

        if should_requery:
            lower, upper = self._query_transformer(study, search_space)

            # Joint hypervolume collapse guard — expand if volume is too small
            lower, upper = self._apply_joint_hypervolume_guard(lower, upper)

            self._cached_lower = lower
            self._cached_upper = upper
            self._last_requery_trial = n_completed
            self._n_requery_count += 1

            # P0: Adaptive trust decay or static decay
            if self._adaptive_trust:
                self._current_trust_factor = self._compute_adaptive_trust(study)
            else:
                self._current_trust_factor = max(
                    self._min_trust_factor,
                    self._current_trust_factor * self._trust_decay,
                )

        # Phase 3: Constrain search space and delegate to TPE
        assert self._cached_lower is not None and self._cached_upper is not None
        assert self._encoder is not None

        from trip_tpe.utils.search_space import apply_region_bounds

        constrained_space = apply_region_bounds(
            search_space=search_space,
            encoder=self._encoder,
            lower_bounds=self._cached_lower,
            upper_bounds=self._cached_upper,
        )

        # Sample from constrained space using TPE
        return self._tpe_sampler.sample_relative(study, trial, constrained_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        name: str,
        distribution: BaseDistribution,
    ) -> Any:
        """Sample a single parameter independently.

        Falls back to TPE's independent sampling.
        """
        return self._tpe_sampler.sample_independent(
            study, trial, name, distribution
        )

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """Called after each trial completes.

        Delegates to TPE for internal state updates.
        """
        self._tpe_sampler.after_trial(study, trial, state, values)

    @property
    def is_model_loaded(self) -> bool:
        """Whether the Transformer model is loaded and active."""
        return self._model is not None

    @property
    def current_trust_factor(self) -> float:
        """Current trust factor for region proposals."""
        return self._current_trust_factor

    @property
    def n_requery_count(self) -> int:
        """Number of times the Transformer has been queried."""
        return self._n_requery_count
