"""
TRIP-TPE: Trajectory-Informed Region Proposal + Tree-structured Parzen Estimator.

A hybrid meta-learning framework that synthesizes Transformer-based region proposals
with Optuna's TPE for highly efficient hyperparameter optimization.

Architecture:
    - Neural Component: Lightweight Transformer (~10M params) trained on HPO trajectories
      to predict promising search space regions from partial optimization history.
    - Optimization Component: Optuna's TPE constrained to search within dynamically
      generated bounds from the Transformer's region proposals.

Key Innovation:
    Unlike PFNs4BO (replaces entire surrogate) or NAP (replaces acquisition function),
    TRIP-TPE isolates the neural component for *macro-region definition* while preserving
    TPE's mathematically rigorous kernel density estimators for fine-grained exploitation.
"""

__version__ = "0.1.0"
__author__ = "Ionuț"

from trip_tpe.samplers.trip_tpe_sampler import TRIPTPESampler

__all__ = ["TRIPTPESampler"]
