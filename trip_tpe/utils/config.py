"""
Configuration management for TRIP-TPE.

Provides a hierarchical configuration system with YAML loading, defaults,
and runtime overrides. All architectural hyperparameters are centralized here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass-based configuration hierarchy
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Configuration for the Region Proposal Transformer.

    Default architecture: ~10M parameters, 4 layers, 256 embedding dim.
    Aligned with the feasibility analysis confirming <150 MB static VRAM
    footprint (FP16) and total usage within 2-3 GB on a GTX 1650 Ti.
    """
    num_layers: int = 4
    embed_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 200  # HPO trajectories rarely exceed 200 steps
    activation: str = "gelu"
    # Input projection dimensions
    hp_input_dim: int = 16   # max hyperparameter dimensions per trial
    obj_input_dim: int = 1   # number of objective values per trial
    # Region proposal head
    region_output_mode: str = "beta"  # "beta" | "gaussian" | "uniform_bounds"
    # --- P1: DETR-style cross-attention dimension queries ---
    # When True, replaces the [CLS] bottleneck with per-dimension learnable
    # queries that cross-attend to the encoded trajectory.
    use_dimension_queries: bool = True
    # Number of cross-attention layers for dimension queries
    dim_query_layers: int = 2
    # --- P2: Mixture-of-Betas output head ---
    # Number of Beta mixture components per dimension (3 = captures multimodal optima)
    n_beta_components: int = 3
    # --- Meta-feature conditioning for cold-start ---
    # Number of dataset-level meta-features (0 = disabled).
    # 10 = Tier 1 (7) + Tier 2 (3):
    #   Tier 1: log_n_samples, log_n_features, n_classes/20,
    #           dimensionality_ratio, imbalance_ratio, frac_categorical, frac_missing
    #   Tier 2: intrinsic_dim_ratio, mean_mutual_info, landmark_1nn_accuracy
    meta_dim: int = 10
    # Whether to use meta-features during training and inference
    use_meta_features: bool = True
    # Meta-feature dropout probability during training.
    # Randomly zeros the entire meta-feature vector for this fraction of samples,
    # forcing the model to also learn without meta-features (prevents over-reliance
    # and improves robustness, analogous to classifier-free guidance in diffusion).
    meta_feature_dropout: float = 0.15


@dataclass
class TrainingConfig:
    """Configuration for training the region proposal model."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    fp16: bool = True  # Mixed precision for VRAM efficiency
    # Curriculum: gradually increase trajectory length during training
    curriculum: bool = True
    curriculum_start_len: int = 10
    curriculum_end_len: int = 100
    curriculum_warmup_epochs: int = 20
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    # Checkpointing
    save_every: int = 10
    eval_every: int = 5
    patience: int = 15  # Early stopping patience


@dataclass
class RegionProposalConfig:
    """Configuration for region proposal behavior at inference time."""
    # Number of initial random trials before querying the Transformer
    n_warmup_trials: int = 5
    # How often to re-query the Transformer (in trial steps)
    requery_interval: int = 10
    # Confidence threshold for region bounds (controls tightness)
    confidence_level: float = 0.9
    # Minimum region width as fraction of full search space per dimension
    min_region_fraction: float = 0.05
    # Maximum region width — prevents overly broad proposals
    max_region_fraction: float = 0.8
    # Blending factor: how much to trust Transformer vs. full space
    # 1.0 = fully trust Transformer bounds, 0.0 = ignore Transformer
    trust_factor: float = 0.8
    # Decay the trust factor over time to allow TPE more freedom
    trust_decay: float = 0.995
    min_trust_factor: float = 0.3
    # Weight for Beta quantile bounds in the ensemble prediction
    # 1.0 = only Beta quantiles, 0.0 = only direct bounds, 0.5 = equal blend
    beta_blend_weight: float = 0.5
    # --- P0: Adaptive trust scheduling with Bayesian regret feedback ---
    # When True, trust_decay is modulated by how well the Transformer's
    # proposals actually cover the top-performing trials.
    adaptive_trust: bool = True
    # Target coverage rate (fraction of top-gamma trials inside proposed region)
    adaptive_trust_target_coverage: float = 0.8
    # Sensitivity of trust adjustment to coverage deviation
    adaptive_trust_sensitivity: float = 5.0
    # Gamma quantile for defining "good" trials in adaptive trust
    adaptive_trust_gamma: float = 0.15


@dataclass
class TPEConfig:
    """Configuration for the underlying TPE sampler."""
    n_startup_trials: int = 0  # We handle warm-start via Transformer
    n_ei_candidates: int = 24
    gamma_strategy: str = "default"  # "default" | "fixed"
    gamma_value: float = 0.15  # Optimal per Watanabe 2023 ablation
    multivariate: bool = True
    group: bool = True
    constant_liar: bool = False
    seed: Optional[int] = None


@dataclass
class EvalConfig:
    """Configuration for benchmarking and evaluation."""
    budgets: List[int] = field(default_factory=lambda: [25, 50, 100])
    n_seeds: int = 20
    benchmarks: List[str] = field(
        default_factory=lambda: ["hpob", "yahpo"]
    )
    baselines: List[str] = field(
        default_factory=lambda: [
            "random", "tpe", "gp_ei", "smac3", "bohb", "hebo", "pfns4bo"
        ]
    )
    metric: str = "normalized_regret"
    output_dir: str = "results"


@dataclass
class TRIPTPEConfig:
    """Top-level configuration aggregating all sub-configs."""
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    region_proposal: RegionProposalConfig = field(default_factory=RegionProposalConfig)
    tpe: TPEConfig = field(default_factory=TPEConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    # Global settings
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    # Weights & Biases
    wandb_project: str = "trip-tpe"
    wandb_entity: Optional[str] = None  # None = default user account
    wandb_run_name: Optional[str] = None  # None = auto-generated


# ---------------------------------------------------------------------------
# YAML serialization / deserialization
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dictionaries."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    """Recursively update a dataclass from a dictionary."""
    for key, value in updates.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict] = None) -> TRIPTPEConfig:
    """Load configuration from YAML file with optional overrides.

    Args:
        path: Path to YAML config file. If None, returns defaults.
        overrides: Dictionary of overrides applied after YAML loading.

    Returns:
        Fully resolved TRIPTPEConfig instance.
    """
    config = TRIPTPEConfig()

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            _update_dataclass(config, yaml_data)

    if overrides:
        _update_dataclass(config, overrides)

    return config


def save_config(config: TRIPTPEConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_dataclass_to_dict(config), f, default_flow_style=False, sort_keys=False)
