"""
PyTorch Dataset for HPO trajectory training pairs.

Handles padding, batching, and collation of variable-length trajectories
into fixed-size tensors suitable for the Transformer encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from trip_tpe.data.preprocessing import META_FEATURE_DIM, TrajectoryPair


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Transformer region proposal training.

    Each item contains:
        - input_seq: Concatenated [config || objective] tokens, shape (max_len, feat_dim)
        - attention_mask: Binary mask, shape (max_len,), 1 = valid token
        - target_lower: Region lower bounds, shape (n_dims,)
        - target_upper: Region upper bounds, shape (n_dims,)
    """

    def __init__(
        self,
        pairs: List[TrajectoryPair],
        max_seq_len: int = 200,
        hp_dim: int = 16,
    ):
        """Initialize the dataset.

        Args:
            pairs: List of preprocessed TrajectoryPair instances.
            max_seq_len: Maximum sequence length (pad/truncate to this).
            hp_dim: Fixed hyperparameter dimensionality (pad shorter spaces).
        """
        self.pairs = pairs
        self.max_seq_len = max_seq_len
        self.hp_dim = hp_dim

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        seq_len = min(pair.seq_len, self.max_seq_len)
        actual_n_dims = pair.input_configs.shape[1]
        pad_dims = self.hp_dim - actual_n_dims

        # Pad hyperparameter dimensions if needed
        configs = pair.input_configs[:seq_len]
        if pad_dims > 0:
            configs = np.pad(configs, ((0, 0), (0, pad_dims)), constant_values=0.0)
        elif pad_dims < 0:
            configs = configs[:, :self.hp_dim]

        objectives = pair.input_objectives[:seq_len]

        # Concatenate configs and objectives: shape (seq_len, hp_dim + 1)
        feat_dim = self.hp_dim + 1
        input_seq = np.concatenate([configs, objectives], axis=1).astype(np.float32)

        # Pad sequence to max_seq_len
        padded = np.zeros((self.max_seq_len, feat_dim), dtype=np.float32)
        padded[:seq_len] = input_seq
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        # Target bounds — pad to hp_dim
        target_lower = np.zeros(self.hp_dim, dtype=np.float32)
        target_upper = np.ones(self.hp_dim, dtype=np.float32)
        n = min(actual_n_dims, self.hp_dim)
        target_lower[:n] = pair.target_lower[:n]
        target_upper[:n] = pair.target_upper[:n]

        # Dimension mask: which dims are "real" (not padding)
        dim_mask = np.zeros(self.hp_dim, dtype=np.float32)
        dim_mask[:min(actual_n_dims, self.hp_dim)] = 1.0

        # Meta-features: use pair's meta_features if available, else zeros
        if pair.meta_features is not None:
            meta = pair.meta_features.astype(np.float32)
            # Pad or truncate to META_FEATURE_DIM
            if len(meta) < META_FEATURE_DIM:
                meta = np.pad(meta, (0, META_FEATURE_DIM - len(meta)))
            elif len(meta) > META_FEATURE_DIM:
                meta = meta[:META_FEATURE_DIM]
        else:
            meta = np.zeros(META_FEATURE_DIM, dtype=np.float32)

        return {
            "input_seq": torch.from_numpy(padded),
            "attention_mask": torch.from_numpy(mask),
            "target_lower": torch.from_numpy(target_lower),
            "target_upper": torch.from_numpy(target_upper),
            "dim_mask": torch.from_numpy(dim_mask),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "meta_features": torch.from_numpy(meta),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader.

        Args:
            batch: List of dataset items.

        Returns:
            Batched tensors with proper stacking.
        """
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0].keys()
        }


class SyntheticTrajectoryDataset(Dataset):
    """Generates synthetic HPO trajectories for pre-training or testing.

    Produces trajectories from known analytical functions where the
    optimal region is known a priori, enabling supervised training
    without requiring HPO-B or YAHPO Gym data.
    """

    def __init__(
        self,
        n_trajectories: int = 10000,
        n_dims: int = 8,
        max_trials: int = 100,
        hp_dim: int = 16,
        max_seq_len: int = 200,
        seed: int = 42,
    ):
        """Initialize the synthetic dataset generator.

        Args:
            n_trajectories: Number of synthetic trajectories to generate.
            n_dims: Number of hyperparameter dimensions per task.
            max_trials: Maximum evaluations per trajectory.
            hp_dim: Fixed HP dim for padding.
            max_seq_len: Max sequence length for padding.
            seed: Random seed.
        """
        self.n_trajectories = n_trajectories
        self.n_dims = n_dims
        self.max_trials = max_trials
        self.hp_dim = hp_dim
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        # Pre-generate all trajectories
        self.data = self._generate_all()

    def _generate_all(self) -> List[Dict[str, torch.Tensor]]:
        """Generate all synthetic trajectories.

        Uses a diverse mixture of four objective function families to
        prevent the model from overfitting to a single landscape type:
            - Quadratic bowls (convex, unimodal)
            - Rosenbrock-like valleys (narrow ridges)
            - Ackley-like functions (multi-modal with global structure)
            - Rastrigin-like functions (highly multi-modal)
        """
        from trip_tpe.data.preprocessing import TrajectoryPreprocessor

        preprocessor = TrajectoryPreprocessor(seed=self.rng.randint(0, 2**31))
        all_items = []

        func_types = ["quadratic", "rosenbrock", "ackley", "rastrigin"]

        for i in range(self.n_trajectories):
            func_type = func_types[i % len(func_types)]
            center = self.rng.uniform(0.15, 0.85, size=self.n_dims)
            scales = self.rng.uniform(0.5, 5.0, size=self.n_dims)

            # Generate random configurations
            n_trials = self.rng.randint(20, self.max_trials + 1)
            configs = self.rng.uniform(0.0, 1.0, size=(n_trials, self.n_dims)).astype(np.float32)

            # Compute objectives using the selected function family
            diff = configs - center[np.newaxis, :]
            objectives = self._compute_objectives(diff, scales, func_type)
            objectives += self.rng.normal(0, 0.01, size=n_trials).astype(np.float32)

            pairs = preprocessor.process_trajectory(
                configs=configs,
                objectives=objectives,
                search_space_id=f"synthetic_{func_type}_{i}",
                minimize=True,
            )

            ds = TrajectoryDataset(pairs, self.max_seq_len, self.hp_dim)
            for j in range(len(ds)):
                all_items.append(ds[j])

        return all_items

    @staticmethod
    def _compute_objectives(
        diff: np.ndarray,
        scales: np.ndarray,
        func_type: str,
    ) -> np.ndarray:
        """Compute objective values for a given function type.

        Args:
            diff: Centered configs (configs - center), shape (n_trials, n_dims).
            scales: Per-dimension scales, shape (n_dims,).
            func_type: One of "quadratic", "rosenbrock", "ackley", "rastrigin".

        Returns:
            Objective values, shape (n_trials,).
        """
        n_dims = diff.shape[1]

        if func_type == "quadratic":
            return np.sum(scales[np.newaxis, :] * diff**2, axis=1)

        elif func_type == "rosenbrock":
            x = diff * 4.0
            objectives = np.zeros(len(diff))
            for d in range(n_dims - 1):
                objectives += 100.0 * (x[:, d + 1] - x[:, d]**2)**2 + (1.0 - x[:, d])**2
            return objectives / max(n_dims, 1)

        elif func_type == "ackley":
            x = diff * 4.0
            sum_sq = np.mean(x**2, axis=1)
            sum_cos = np.mean(np.cos(2.0 * np.pi * x), axis=1)
            return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20.0 + np.e

        elif func_type == "rastrigin":
            x = diff * 5.12
            return (10.0 * n_dims + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)) / max(n_dims, 1)

        else:
            # Fallback to quadratic
            return np.sum(scales[np.newaxis, :] * diff**2, axis=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]
