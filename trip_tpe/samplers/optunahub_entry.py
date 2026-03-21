"""
OptunaHub plugin entry point for TRIP-TPE.

This module provides the interface required for OptunaHub integration,
enabling TRIP-TPE to be installed and used as a pip-installable plugin:

    pip install trip-tpe

    import optuna
    from trip_tpe import TRIPTPESampler

    sampler = TRIPTPESampler(model_path="path/to/checkpoint.pt")
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

OptunaHub registration (pyproject.toml):
    [project.entry-points."optuna.samplers"]
    TRIPTPESampler = "trip_tpe.samplers.trip_tpe_sampler:TRIPTPESampler"
"""

from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler

__all__ = ["TRIPTPESampler"]

# OptunaHub metadata
SAMPLER_NAME = "TRIPTPESampler"
SAMPLER_DESCRIPTION = (
    "Trajectory-Informed Region Proposal + TPE: A hybrid meta-learning sampler "
    "that uses a pre-trained Transformer to propose promising search space regions, "
    "then delegates fine-grained sampling to TPE within those constrained regions."
)
SAMPLER_REFERENCE = "https://github.com/trip-tpe/trip-tpe"
SAMPLER_AUTHORS = ["Ionuț"]
SAMPLER_VERSION = "0.1.0"
