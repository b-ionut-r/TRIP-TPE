"""
TRIP-TPE Sampler: The core Optuna sampler integrating Transformer region proposals with TPE.

This is the primary user-facing component — a drop-in replacement for Optuna's TPESampler
that augments standard TPE with learned guidance from a pre-trained Transformer.

Two operational modes:

  "guided" (default, recommended):
      Uses the Transformer to SAMPLE promising configurations during warmup,
      giving TPE high-quality seed data. After warmup, periodically injects
      Transformer-guided samples among TPE's own. TPE always operates on the
      FULL search space — no hard constraints. This has a symmetric risk
      profile: bad Transformer suggestions waste a few trials but never lock
      out the optimum.

  "constrained" (legacy):
      The original approach: constrain TPE's search space to the Transformer's
      proposed region. Kept for backward compatibility and ablation. NOT
      recommended for production use due to asymmetric risk (wrong regions
      are catastrophic, right regions give marginal benefit).

Key Design Decisions:
    - In guided mode, the Transformer contributes through EXPLORATION (sampling),
      not EXPLOITATION (constraining). This mirrors how meta-learning actually
      helps in BO: better initialization, not tighter bounds.
    - TPE handles ALL density estimation on the full search space.
    - Bad Transformer samples are just data points that TPE naturally down-weights
      in its good/bad split — they cannot lock out regions.
"""

from __future__ import annotations

import copy
import math
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

    A hybrid sampler that uses a pre-trained Transformer to guide exploration,
    then delegates fine-grained optimization to TPE on the full search space.

    Example (guided mode — recommended):
        >>> import optuna
        >>> from trip_tpe import TRIPTPESampler
        >>>
        >>> sampler = TRIPTPESampler(
        ...     model_path="checkpoints/trip_tpe_best.pt",
        ...     mode="guided",
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
        # Mode selection
        mode: str = "guided",
        # Guided mode parameters — strong Transformer involvement
        n_guided_exploration: Union[int, float] = 0.3,
        inject_rate: float = 0.4,
        inject_decay: float = 0.99,
        min_inject_rate: float = 0.10,
        exploration_temperature: float = 0.15,
        # Burst re-centering: periodically fire consecutive Transformer samples
        burst_requery_interval: int = 20,
        burst_size: int = 2,
        # Budget hint (optional): if known, used to resolve fractional n_guided_exploration
        n_trials_hint: Optional[int] = None,
        # Legacy constrained mode parameters
        n_warmup_trials: int = 5,
        requery_interval: int = 10,
        confidence_level: float = 0.9,
        min_region_fraction: float = 0.05,
        max_region_fraction: float = 0.8,
        trust_factor: float = 0.8,
        trust_decay: float = 0.995,
        min_trust_factor: float = 0.3,
        beta_blend_weight: float = 0.5,
        adaptive_trust: bool = False,
        adaptive_trust_target_coverage: float = 0.8,
        adaptive_trust_sensitivity: float = 5.0,
        adaptive_trust_gamma: float = 0.15,
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
        # Meta-features for cold-start conditioning
        meta_features: Optional[np.ndarray] = None,
    ):
        """Initialize the TRIP-TPE sampler.

        Args:
            model_path: Path to pre-trained Transformer checkpoint (.pt file).
                        If None, operates as enhanced TPE without Transformer guidance.
            mode: "guided" (recommended) or "constrained" (legacy).
                  Guided mode samples FROM the Transformer's distributions.
                  Constrained mode restricts TPE's search space (legacy, not recommended).
            n_guided_exploration: Transformer-guided seed trials before TPE takes over.
                                  If float in (0,1): fraction of n_trials (resolved dynamically
                                  or via n_trials_hint). Clamped to [5, n_trials*0.6].
                                  If int >= 1: absolute number (used as-is).
                                  Default 0.4 = 40% of budget. Only used in guided mode.
            inject_rate: Fraction of post-warmup trials that use Transformer guidance
                         instead of TPE. Decays over time. Only used in guided mode.
            inject_decay: Multiplicative decay per trial for inject_rate.
            min_inject_rate: Floor for inject_rate after decay.
            exploration_temperature: Softening factor for Transformer sampling (0-1).
                                     0 = exact model distribution, 1 = uniform.
                                     Blends Beta params toward Beta(1,1).
            burst_requery_interval: Every N TPE trials, trigger a burst of consecutive
                                    Transformer samples. Re-centers exploration as the
                                    trajectory evolves. Only used in guided mode.
            burst_size: Number of consecutive Transformer samples in each burst.
            n_warmup_trials: Random warmup trials (constrained mode only).
            requery_interval: Re-query interval (constrained mode only).
            confidence_level: Beta quantile confidence (constrained mode only).
            min_region_fraction: Min region width (constrained mode only).
            max_region_fraction: Max region width (constrained mode only).
            trust_factor: Initial trust in proposals (constrained mode only).
            trust_decay: Trust decay rate (constrained mode only).
            min_trust_factor: Trust floor (constrained mode only).
            beta_blend_weight: Beta/direct blend weight (constrained mode only).
            adaptive_trust: Use adaptive trust (constrained mode only).
            n_startup_trials: TPE startup trials (0 = TPE uses KDE immediately).
            n_ei_candidates: Number of EI candidates for TPE.
            gamma: TPE gamma quantile. None = Optuna default.
            multivariate: Use multivariate TPE.
            group: Use group decomposition in TPE.
            constant_liar: Use constant liar for parallel sampling.
            seed: Random seed.
            device: Device for Transformer inference ("auto", "cuda", "cpu").
            hp_dim: Maximum hyperparameter dimensions the model supports.
            max_seq_len: Maximum trajectory sequence length for Transformer input.
            meta_features: Optional dataset-level meta-features for cold-start
                          conditioning. Shape (META_FEATURE_DIM,) = (10,).
        """
        if mode not in ("guided", "constrained"):
            raise ValueError(f"mode must be 'guided' or 'constrained', got '{mode}'")

        self._mode = mode
        self._model_path = model_path
        self._hp_dim = hp_dim
        self._max_seq_len = max_seq_len
        self._seed = seed
        self._meta_features = meta_features

        # Guided mode state
        # n_guided_exploration: if float, resolve dynamically; if int, use directly
        if isinstance(n_guided_exploration, float) and 0 < n_guided_exploration < 1:
            self._guided_fraction = n_guided_exploration
            if n_trials_hint is not None and n_trials_hint > 0:
                raw = int(n_trials_hint * n_guided_exploration)
                self._n_guided_exploration = max(5, min(raw, int(n_trials_hint * 0.6)))
                self._guided_resolved = True
            else:
                # Will resolve on first sample_relative call
                self._n_guided_exploration = 20  # temporary fallback
                self._guided_resolved = False
        else:
            self._guided_fraction = None
            self._n_guided_exploration = int(n_guided_exploration)
            self._guided_resolved = True

        self._inject_rate = inject_rate
        self._inject_decay = inject_decay
        self._min_inject_rate = min_inject_rate
        self._exploration_temperature = exploration_temperature
        self._burst_requery_interval = burst_requery_interval
        self._burst_size = burst_size
        self._burst_remaining = 0  # countdown for active burst
        self._last_burst_trial = 0  # trial number of last burst trigger
        self._trial_sources: Dict[int, str] = {}  # trial_number → "transformer"|"tpe"
        self._rng = np.random.RandomState(seed)

        # Legacy constrained mode state
        self._n_warmup_trials = n_warmup_trials
        self._requery_interval = requery_interval
        self._confidence_level = confidence_level
        self._min_region_fraction = min_region_fraction
        self._max_region_fraction = max_region_fraction
        self._trust_factor_init = trust_factor
        self._trust_decay = trust_decay
        self._min_trust_factor = min_trust_factor
        self._beta_blend_weight = beta_blend_weight
        self._adaptive_trust = adaptive_trust
        self._adaptive_trust_target_coverage = adaptive_trust_target_coverage
        self._adaptive_trust_sensitivity = adaptive_trust_sensitivity
        self._adaptive_trust_gamma = adaptive_trust_gamma
        self._current_trust_factor = trust_factor
        self._last_requery_trial = 0
        self._cached_lower: Optional[np.ndarray] = None
        self._cached_upper: Optional[np.ndarray] = None
        self._n_requery_count = 0

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
        self._tpe_sampler = TPESampler(**tpe_kwargs)

        # Model state (lazy-loaded)
        self._model: Optional[Any] = None
        self._model_loaded = False
        self._encoder: Optional[Any] = None
        self._search_space_built = False

    # ================================================================
    # Model & encoder management (shared between modes)
    # ================================================================

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

            if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                config = checkpoint["model_config"]
                self._model = RegionProposalTransformer(**config)
                self._model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model = RegionProposalTransformer(hp_dim=self._hp_dim)
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
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
        """Build the search space encoder on first invocation."""
        if self._search_space_built:
            return

        from trip_tpe.utils.search_space import SearchSpaceEncoder
        self._encoder = SearchSpaceEncoder(search_space)
        self._search_space_built = True

    def _encode_trajectory(
        self, study: Study, search_space: Dict[str, BaseDistribution]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Encode the current optimization history as a Transformer-ready sequence.

        Returns:
            Tuple of (input_seq, attention_mask, actual_len).
            input_seq: shape (1, max_seq_len, hp_dim + 1)
            attention_mask: shape (1, max_seq_len)
        """
        assert self._encoder is not None

        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]

        if not completed:
            return (
                np.zeros((1, 1, self._hp_dim + 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                0,
            )

        completed = sorted(completed, key=lambda t: t.number)

        configs = []
        objectives = []
        _warned_multi_obj = False
        for trial in completed:
            encoded = self._encoder.encode_params(trial.params)
            padded = np.zeros(self._hp_dim, dtype=np.float32)
            n = min(len(encoded), self._hp_dim)
            padded[:n] = encoded[:n]
            configs.append(padded)

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

        # Rank-normalize objectives (0 = best, 1 = worst)
        n = len(obj_arr)
        if n > 1:
            direction = getattr(study, "direction", None)
            if direction is not None and direction == optuna.study.StudyDirection.MAXIMIZE:
                ranks = np.argsort(np.argsort(-obj_arr)).astype(np.float32)
            else:
                ranks = np.argsort(np.argsort(obj_arr)).astype(np.float32)
            norm_obj = (ranks / (n - 1)).reshape(-1, 1)
        else:
            norm_obj = np.array([[0.5]], dtype=np.float32)

        seq = np.concatenate([configs_arr, norm_obj], axis=1)

        max_len = self._max_seq_len
        actual_len = min(n, max_len)
        seq = seq[:actual_len]

        padded_seq = np.zeros((1, max_len, self._hp_dim + 1), dtype=np.float32)
        padded_seq[0, :actual_len] = seq
        mask = np.zeros((1, max_len), dtype=np.float32)
        mask[0, :actual_len] = 1.0

        return padded_seq, mask, actual_len

    # ================================================================
    # Optuna BaseSampler interface
    # ================================================================

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        """Infer the search space for relative sampling. Delegates to TPE."""
        return self._tpe_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Sample parameters using Transformer-guided TPE.

        This is the main sampling method called by Optuna. Dispatches to
        the appropriate mode (guided or constrained).
        """
        if not search_space:
            return {}

        # Lazy-resolve fractional n_guided_exploration from study budget
        if not self._guided_resolved and self._guided_fraction is not None:
            # Try to infer n_trials from Optuna's internal state
            n_trials_budget = getattr(study, '_n_trials', None)
            if n_trials_budget is None:
                # Fallback: assume 100 trials (conservative default)
                n_trials_budget = 100
            raw = int(n_trials_budget * self._guided_fraction)
            self._n_guided_exploration = max(5, min(raw, int(n_trials_budget * 0.6)))
            self._guided_resolved = True

        n_completed = len([
            t for t in study.trials if t.state == TrialState.COMPLETE
        ])

        # Build encoder and load model
        self._build_encoder(search_space)
        has_model = self._load_model()

        if not has_model:
            return self._tpe_sampler.sample_relative(study, trial, search_space)

        if self._mode == "guided":
            return self._sample_guided(study, trial, search_space, n_completed)
        else:
            return self._sample_constrained(study, trial, search_space, n_completed)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        name: str,
        distribution: BaseDistribution,
    ) -> Any:
        """Sample a single parameter independently. Falls back to TPE."""
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
        """Called after each trial completes."""
        # Tag the trial with its source for analysis
        if trial.number in self._trial_sources:
            try:
                study._storage.set_trial_user_attr(
                    trial._trial_id, "trip_source", self._trial_sources[trial.number]
                )
            except Exception:
                pass  # Non-critical — don't break the optimization loop

        self._tpe_sampler.after_trial(study, trial, state, values)

    # ================================================================
    # GUIDED MODE — the recommended approach
    # ================================================================

    def _sample_guided(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
        n_completed: int,
    ) -> Dict[str, Any]:
        """Guided mode: Transformer samples seed trials, then TPE on full space.

        Phase 1 (trials 0 to n_guided_exploration - 1):
            Sample configs from the Transformer's Beta distributions.
            These become high-quality seed points for TPE's density estimation.

        Phase 2 (trials n_guided_exploration onwards):
            TPE operates on the FULL search space with two Transformer
            re-injection mechanisms:
              a) Stochastic injection: each trial has a decaying probability of
                 being a Transformer sample (inject_rate * decay^n).
              b) Burst re-centering: every `burst_requery_interval` TPE trials,
                 fire `burst_size` consecutive Transformer samples. This ensures
                 the Transformer periodically re-evaluates the trajectory and
                 injects a fresh batch of guided exploration, keeping TPE's
                 density estimation anchored to promising regions even in late
                 optimization stages.
        """
        # Phase 1: Guided exploration (replaces random warmup)
        if n_completed < self._n_guided_exploration:
            params = self._sample_from_transformer(study, search_space)
            if params is not None:
                self._trial_sources[trial.number] = "transformer"
                return params
            # Fallback to TPE if Transformer sampling fails
            self._trial_sources[trial.number] = "tpe"
            return self._tpe_sampler.sample_relative(study, trial, search_space)

        # Phase 2: TPE with Transformer injection + burst re-centering

        # Check burst: are we in the middle of an active burst?
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            params = self._sample_from_transformer(study, search_space)
            if params is not None:
                self._trial_sources[trial.number] = "transformer"
                return params

        # Check burst trigger: has enough TPE time passed since last burst?
        n_since_guided = n_completed - self._n_guided_exploration
        n_since_last_burst = n_completed - self._last_burst_trial
        if (
            self._burst_requery_interval > 0
            and n_since_guided >= self._burst_requery_interval
            and n_since_last_burst >= self._burst_requery_interval
        ):
            # Trigger a new burst
            self._burst_remaining = self._burst_size - 1  # -1 because this trial counts
            self._last_burst_trial = n_completed
            params = self._sample_from_transformer(study, search_space)
            if params is not None:
                self._trial_sources[trial.number] = "transformer"
                return params

        # Stochastic injection (decaying probability)
        if self._should_inject(n_completed):
            params = self._sample_from_transformer(study, search_space)
            if params is not None:
                self._trial_sources[trial.number] = "transformer"
                return params

        # Default: pure TPE on FULL search space (no constraints!)
        self._trial_sources[trial.number] = "tpe"
        return self._tpe_sampler.sample_relative(study, trial, search_space)

    def _sample_from_transformer(
        self,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> Optional[Dict[str, Any]]:
        """Sample a configuration from the Transformer's predicted Beta distributions.

        Instead of using the Transformer to constrain TPE's search space, we
        sample directly from the Transformer's output distributions. This produces
        configurations biased toward promising regions without hard constraints.

        Uses temperature softening to blend the model's predictions toward
        uniform, ensuring adequate exploration diversity.

        Returns:
            Parameter dictionary in the search space's native range, or None
            if sampling fails.
        """
        assert self._model is not None and self._encoder is not None

        try:
            # 1. Encode the current trajectory
            input_seq, mask, actual_len = self._encode_trajectory(study, search_space)

            # 2. Prepare tensors for the model
            input_tensor = torch.from_numpy(input_seq).to(self._device)
            mask_tensor = torch.from_numpy(mask).to(self._device)

            n_dims = self._encoder.n_dims
            dim_mask = np.zeros((1, self._hp_dim), dtype=np.float32)
            dim_mask[0, :min(n_dims, self._hp_dim)] = 1.0
            dim_mask_tensor = torch.from_numpy(dim_mask).to(self._device)

            meta_tensor = None
            if self._meta_features is not None:
                meta_np = np.array(self._meta_features, dtype=np.float32).reshape(1, -1)
                meta_tensor = torch.from_numpy(meta_np).to(self._device)

            # 3. Forward pass — get raw Beta parameters
            with torch.no_grad():
                outputs = self._model(input_tensor, mask_tensor, meta_features=meta_tensor)

            alpha = outputs["alpha"]        # (1, hp_dim) or (1, hp_dim, K)
            beta_param = outputs["beta"]    # same shape
            mix_weights = outputs.get("mix_weights")  # (1, hp_dim, K) or None

            # 4. Sample from Beta distributions with temperature softening
            temp = self._exploration_temperature
            is_mixture = alpha.dim() == 3 and alpha.shape[-1] > 1

            if is_mixture and mix_weights is not None:
                # Mixture of Betas: select component per dimension, then sample
                K = alpha.shape[-1]
                a_all = alpha[0, :n_dims].cpu().numpy().astype(np.float64)
                b_all = beta_param[0, :n_dims].cpu().numpy().astype(np.float64)
                w_all = mix_weights[0, :n_dims].cpu().numpy().astype(np.float64)

                # Select component per dimension via categorical sampling
                selected_k = np.array([
                    self._rng.choice(K, p=np.clip(w_all[d], 0, None) / np.clip(w_all[d], 0, None).sum())
                    for d in range(n_dims)
                ])

                # Gather selected component parameters
                a_sel = a_all[np.arange(n_dims), selected_k]
                b_sel = b_all[np.arange(n_dims), selected_k]
            else:
                # Single Beta per dimension
                a_sel = alpha[0, :n_dims].cpu().numpy().astype(np.float64)
                b_sel = beta_param[0, :n_dims].cpu().numpy().astype(np.float64)

            # Temperature softening: blend toward Beta(1,1) = Uniform[0,1]
            # At temp=0: exact model distribution
            # At temp=1: uniform distribution
            a_soft = a_sel * (1.0 - temp) + 1.0 * temp
            b_soft = b_sel * (1.0 - temp) + 1.0 * temp

            # Numerical safety: ensure valid Beta parameters
            a_soft = np.maximum(a_soft, 0.05)
            b_soft = np.maximum(b_soft, 0.05)

            # Vectorized Beta sampling
            samples = self._rng.beta(a_soft, b_soft)

            # 5. Decode from [0,1] normalized space to parameter space
            encoded_full = np.zeros(self._hp_dim, dtype=np.float32)
            encoded_full[:n_dims] = samples.astype(np.float32)
            params = self._encoder.decode_params(encoded_full)

            # 6. Filter to only parameters in the current search space
            return {k: v for k, v in params.items() if k in search_space}

        except Exception as e:
            warnings.warn(
                f"TRIP-TPE Transformer sampling failed: {e}. Falling back to TPE.",
                RuntimeWarning,
            )
            return None

    def _should_inject(self, n_completed: int) -> bool:
        """Decide whether to inject a Transformer-guided sample.

        Uses exponentially decaying injection rate. Early after warmup,
        the Transformer still contributes frequently. As TPE accumulates
        data and becomes self-sufficient, injections fade to near zero.

        Args:
            n_completed: Number of completed trials so far.

        Returns:
            True if this trial should use Transformer guidance.
        """
        n_since_guided = n_completed - self._n_guided_exploration
        if n_since_guided < 0:
            return True  # Still in guided exploration phase

        # Exponential decay
        rate = self._inject_rate * (self._inject_decay ** n_since_guided)
        rate = max(rate, self._min_inject_rate)

        return self._rng.random() < rate

    # ================================================================
    # CONSTRAINED MODE — legacy approach (kept for ablation)
    # ================================================================

    def _sample_constrained(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
        n_completed: int,
    ) -> Dict[str, Any]:
        """Legacy constrained mode: restrict TPE's search space to Transformer regions.

        WARNING: This mode has an asymmetric risk profile. If the Transformer
        proposes a wrong region, it catastrophically prevents TPE from finding
        the optimum. Use guided mode instead.
        """
        # Phase 1: Warmup — use TPE (which does random at n_startup=0)
        if n_completed < self._n_warmup_trials:
            return self._tpe_sampler.sample_relative(study, trial, search_space)

        # Phase 2: Query Transformer for region proposals (or use cached)
        should_requery = (
            self._cached_lower is None
            or (n_completed - self._last_requery_trial) >= self._requery_interval
        )

        if should_requery:
            lower, upper = self._query_transformer_bounds(study, search_space)
            lower, upper = self._apply_joint_hypervolume_guard(lower, upper)

            self._cached_lower = lower
            self._cached_upper = upper
            self._last_requery_trial = n_completed
            self._n_requery_count += 1

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

        return self._tpe_sampler.sample_relative(study, trial, constrained_space)

    def _query_transformer_bounds(
        self,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the Transformer for region bound proposals (constrained mode).

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (n_dims,).
        """
        assert self._model is not None and self._encoder is not None

        input_seq, mask, actual_len = self._encode_trajectory(study, search_space)

        if actual_len == 0:
            n_dims = self._encoder.n_dims
            return np.zeros(n_dims, dtype=np.float32), np.ones(n_dims, dtype=np.float32)

        input_tensor = torch.from_numpy(input_seq).to(self._device)
        mask_tensor = torch.from_numpy(mask).to(self._device)

        n_dims = self._encoder.n_dims
        dim_mask = np.zeros((1, self._hp_dim), dtype=np.float32)
        dim_mask[0, :min(n_dims, self._hp_dim)] = 1.0
        dim_mask_tensor = torch.from_numpy(dim_mask).to(self._device)

        meta_tensor = None
        if self._meta_features is not None:
            meta_np = np.array(self._meta_features, dtype=np.float32).reshape(1, -1)
            meta_tensor = torch.from_numpy(meta_np).to(self._device)

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

        # Trust factor blending
        tf = self._current_trust_factor
        lower_np = tf * lower_np + (1 - tf) * 0.0
        upper_np = tf * upper_np + (1 - tf) * 1.0

        # Enforce min/max region fractions
        width = upper_np - lower_np
        midpoints = (lower_np + upper_np) / 2.0

        too_narrow = width < self._min_region_fraction
        lower_np[too_narrow] = np.clip(
            midpoints[too_narrow] - self._min_region_fraction / 2, 0.0, 1.0
        )
        upper_np[too_narrow] = np.clip(
            midpoints[too_narrow] + self._min_region_fraction / 2, 0.0, 1.0
        )

        too_wide = width > self._max_region_fraction
        lower_np[too_wide] = np.clip(
            midpoints[too_wide] - self._max_region_fraction / 2, 0.0, 1.0
        )
        upper_np[too_wide] = np.clip(
            midpoints[too_wide] + self._max_region_fraction / 2, 0.0, 1.0
        )

        return lower_np, upper_np

    def _compute_adaptive_trust(self, study: Study) -> float:
        """Coverage-aware trust factor adjustment (constrained mode only)."""
        if self._cached_lower is None or self._cached_upper is None:
            return self._current_trust_factor
        if self._encoder is None:
            return self._current_trust_factor

        completed = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]
        if len(completed) < 3:
            return self._current_trust_factor

        gamma = self._adaptive_trust_gamma
        n_good = max(1, int(len(completed) * gamma))
        values = [t.value for t in completed if t.value is not None]
        if not values:
            return self._current_trust_factor

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

        n_inside = 0
        for trial_obj in good_trials:
            encoded = self._encoder.encode_params(trial_obj.params)
            n_dims = min(len(encoded), len(self._cached_lower))
            inside = all(
                self._cached_lower[d] <= encoded[d] <= self._cached_upper[d]
                for d in range(n_dims)
            )
            if inside:
                n_inside += 1

        coverage_rate = n_inside / max(len(good_trials), 1)

        target = self._adaptive_trust_target_coverage
        k = self._adaptive_trust_sensitivity
        adjustment = 1.0 / (1.0 + math.exp(-k * (coverage_rate - target)))
        decay_multiplier = 0.5 + adjustment
        new_trust = self._current_trust_factor * (self._trust_decay ** decay_multiplier)
        new_trust = max(self._min_trust_factor, new_trust)

        return new_trust

    def _apply_joint_hypervolume_guard(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Guard against joint hypervolume collapse (constrained mode only)."""
        widths = upper - lower
        safe_widths = np.maximum(widths, 1e-12)
        log_volume = np.sum(np.log(safe_widths))

        n_dims = len(widths)
        dimension_aware_threshold = self._min_region_fraction ** n_dims
        log_threshold = np.log(max(dimension_aware_threshold, 1e-300))

        if log_volume >= log_threshold:
            return lower, upper

        if n_dims == 0:
            return lower, upper

        log_deficit = (log_threshold - log_volume) / n_dims
        expansion_factor = np.exp(log_deficit)

        midpoints = (lower + upper) / 2.0
        new_half_widths = safe_widths * expansion_factor / 2.0
        lower_expanded = np.clip(midpoints - new_half_widths, 0.0, 1.0)
        upper_expanded = np.clip(midpoints + new_half_widths, 0.0, 1.0)

        return lower_expanded, upper_expanded

    # ================================================================
    # Properties
    # ================================================================

    @property
    def mode(self) -> str:
        """Current operational mode ('guided' or 'constrained')."""
        return self._mode

    @property
    def is_model_loaded(self) -> bool:
        """Whether the Transformer model is loaded and active."""
        return self._model is not None

    @property
    def current_trust_factor(self) -> float:
        """Current trust factor (constrained mode only)."""
        return self._current_trust_factor

    @property
    def n_requery_count(self) -> int:
        """Number of times the Transformer has been queried."""
        return self._n_requery_count

    @property
    def trial_sources(self) -> Dict[int, str]:
        """Map of trial numbers to their sampling source."""
        return dict(self._trial_sources)
