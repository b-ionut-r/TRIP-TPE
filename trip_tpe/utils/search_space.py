"""
Search space utilities for TRIP-TPE.

Handles the encoding/decoding of Optuna search space distributions into
fixed-dimensional tensors that the Transformer can process, and the
application of region bounds back onto Optuna distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)


@dataclass
class DimensionInfo:
    """Metadata for a single search space dimension."""
    name: str
    dtype: str  # "float", "int", "categorical"
    low: float
    high: float
    log_scale: bool = False
    num_categories: int = 0  # Only for categorical
    step: Optional[float] = None


class SearchSpaceEncoder:
    """Encodes/decodes Optuna search spaces into normalized tensor representations.

    All continuous and integer parameters are normalized to [0, 1].
    Categorical parameters are one-hot encoded and then mapped to [0, 1].
    Log-scale parameters are transformed to log-space before normalization.
    """

    def __init__(self, search_space: Dict[str, optuna.distributions.BaseDistribution]):
        """Initialize encoder from an Optuna search space dictionary.

        Args:
            search_space: Mapping of parameter names to Optuna distributions.
        """
        self.search_space = search_space
        self.dimensions: List[DimensionInfo] = []
        self._build_dimensions()

    def _build_dimensions(self) -> None:
        """Parse the Optuna search space into DimensionInfo objects."""
        for name, dist in self.search_space.items():
            if isinstance(dist, FloatDistribution):
                self.dimensions.append(DimensionInfo(
                    name=name,
                    dtype="float",
                    low=dist.low,
                    high=dist.high,
                    log_scale=dist.log,
                    step=dist.step,
                ))
            elif isinstance(dist, IntDistribution):
                self.dimensions.append(DimensionInfo(
                    name=name,
                    dtype="int",
                    low=float(dist.low),
                    high=float(dist.high),
                    log_scale=dist.log,
                    step=float(dist.step) if dist.step else None,
                ))
            elif isinstance(dist, CategoricalDistribution):
                self.dimensions.append(DimensionInfo(
                    name=name,
                    dtype="categorical",
                    low=0.0,
                    high=float(len(dist.choices) - 1),
                    num_categories=len(dist.choices),
                ))

    @property
    def n_dims(self) -> int:
        """Number of dimensions in the encoded search space."""
        return len(self.dimensions)

    def encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode a parameter dictionary to a normalized [0, 1] vector.

        Args:
            params: Dictionary mapping parameter names to values.

        Returns:
            1D numpy array of shape (n_dims,) with values in [0, 1].
        """
        encoded = np.zeros(self.n_dims, dtype=np.float32)
        for i, dim in enumerate(self.dimensions):
            val = params.get(dim.name)
            if val is None:
                encoded[i] = 0.5  # Default to midpoint for missing params
                continue

            if dim.dtype == "categorical":
                # Normalize category index to [0, 1]
                dist = self.search_space[dim.name]
                assert isinstance(dist, CategoricalDistribution)
                try:
                    idx = list(dist.choices).index(val)
                except ValueError:
                    idx = 0
                encoded[i] = idx / max(dim.num_categories - 1, 1)
            else:
                fval = float(val)
                if dim.log_scale and fval > 0 and dim.low > 0:
                    fval = np.log(fval)
                    low = np.log(dim.low)
                    high = np.log(dim.high)
                else:
                    low = dim.low
                    high = dim.high
                # Normalize to [0, 1]
                if high - low > 1e-12:
                    encoded[i] = (fval - low) / (high - low)
                else:
                    encoded[i] = 0.5
                encoded[i] = np.clip(encoded[i], 0.0, 1.0)

        return encoded

    def decode_params(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode a normalized [0, 1] vector back to parameter values.

        Args:
            encoded: 1D array of shape (n_dims,) with values in [0, 1].

        Returns:
            Dictionary mapping parameter names to decoded values.
        """
        params: Dict[str, Any] = {}
        for i, dim in enumerate(self.dimensions):
            val = float(encoded[i])
            val = np.clip(val, 0.0, 1.0)

            if dim.dtype == "categorical":
                dist = self.search_space[dim.name]
                assert isinstance(dist, CategoricalDistribution)
                idx = int(round(val * (dim.num_categories - 1)))
                idx = np.clip(idx, 0, dim.num_categories - 1)
                params[dim.name] = dist.choices[idx]
            elif dim.dtype == "int":
                if dim.log_scale and dim.low > 0:
                    low = np.log(dim.low)
                    high = np.log(dim.high)
                    decoded = np.exp(low + val * (high - low))
                else:
                    decoded = dim.low + val * (dim.high - dim.low)
                step = dim.step or 1.0
                params[dim.name] = int(round(decoded / step) * step)
            else:
                if dim.log_scale and dim.low > 0:
                    low = np.log(dim.low)
                    high = np.log(dim.high)
                    decoded = np.exp(low + val * (high - low))
                else:
                    decoded = dim.low + val * (dim.high - dim.low)
                params[dim.name] = float(decoded)

        return params

    def encode_bounds(
        self, lower: Dict[str, float], upper: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode region bounds as normalized vectors.

        Args:
            lower: Lower bounds per parameter name.
            upper: Upper bounds per parameter name.

        Returns:
            Tuple of (lower_encoded, upper_encoded), each shape (n_dims,).
        """
        low_enc = np.zeros(self.n_dims, dtype=np.float32)
        high_enc = np.ones(self.n_dims, dtype=np.float32)

        for i, dim in enumerate(self.dimensions):
            if dim.name in lower:
                low_enc[i] = self.encode_params({dim.name: lower[dim.name]})[i]
            if dim.name in upper:
                high_enc[i] = self.encode_params({dim.name: upper[dim.name]})[i]

        return low_enc, high_enc


def apply_region_bounds(
    search_space: Dict[str, optuna.distributions.BaseDistribution],
    encoder: SearchSpaceEncoder,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Dict[str, optuna.distributions.BaseDistribution]:
    """Create a constrained search space from Transformer-proposed region bounds.

    This is the critical function that bridges the Transformer's output
    (normalized [0,1] bounds per dimension) with Optuna's distribution objects.

    Args:
        search_space: Original Optuna search space.
        encoder: The SearchSpaceEncoder used to normalize/denormalize.
        lower_bounds: Normalized lower bounds, shape (n_dims,).
        upper_bounds: Normalized upper bounds, shape (n_dims,).

    Returns:
        New search space dict with constrained distributions.
    """
    constrained: Dict[str, optuna.distributions.BaseDistribution] = {}

    for i, dim in enumerate(encoder.dimensions):
        lo = float(np.clip(lower_bounds[i], 0.0, 1.0))
        hi = float(np.clip(upper_bounds[i], 0.0, 1.0))

        # Ensure lo < hi
        if hi - lo < 1e-6:
            mid = (lo + hi) / 2.0
            lo = max(0.0, mid - 0.025)
            hi = min(1.0, mid + 0.025)

        original_dist = search_space[dim.name]

        if isinstance(original_dist, FloatDistribution):
            if dim.log_scale and dim.low > 0:
                log_low = np.log(dim.low)
                log_high = np.log(dim.high)
                new_low = float(np.exp(log_low + lo * (log_high - log_low)))
                new_high = float(np.exp(log_low + hi * (log_high - log_low)))
            else:
                new_low = dim.low + lo * (dim.high - dim.low)
                new_high = dim.low + hi * (dim.high - dim.low)
            # Respect step constraints
            constrained[dim.name] = FloatDistribution(
                low=max(new_low, dim.low),
                high=min(new_high, dim.high),
                log=original_dist.log,
                step=original_dist.step,
            )

        elif isinstance(original_dist, IntDistribution):
            if dim.log_scale and dim.low > 0:
                log_low = np.log(dim.low)
                log_high = np.log(dim.high)
                new_low = int(np.exp(log_low + lo * (log_high - log_low)))
                new_high = int(np.ceil(np.exp(log_low + hi * (log_high - log_low))))
            else:
                new_low = int(dim.low + lo * (dim.high - dim.low))
                new_high = int(np.ceil(dim.low + hi * (dim.high - dim.low)))
            step = original_dist.step or 1
            new_low = max(new_low, int(dim.low))
            new_high = min(new_high, int(dim.high))
            # Ensure at least one step
            if new_high <= new_low:
                new_high = new_low + step
            constrained[dim.name] = IntDistribution(
                low=new_low,
                high=new_high,
                log=original_dist.log,
                step=original_dist.step,
            )

        elif isinstance(original_dist, CategoricalDistribution):
            # For categorical: keep all choices (region proposals don't constrain categories)
            constrained[dim.name] = original_dist

    return constrained
