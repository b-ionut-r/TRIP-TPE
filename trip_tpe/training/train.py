"""
Training script for the TRIP-TPE Region Proposal Transformer.

Implements the full training pipeline:
    1. Data loading (synthetic or HPO-B trajectories)
    2. Model initialization with architecture from config
    3. Mixed-precision (FP16) training with AdamW + cosine annealing
    4. Curriculum learning: gradually increase trajectory prefix lengths
    5. Validation with region coverage and tightness metrics
    6. Checkpointing and early stopping

Usage:
    trip-tpe-train --config configs/default.yaml
    trip-tpe-train --synthetic --n-trajectories 50000 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from tqdm import tqdm

from trip_tpe.data.preprocessing import TrajectoryPreprocessor
from trip_tpe.data.trajectory_dataset import TrajectoryDataset, SyntheticTrajectoryDataset
from trip_tpe.models.region_proposal_transformer import RegionProposalTransformer
from trip_tpe.training.loss import RegionProposalLoss
from trip_tpe.utils.config import TRIPTPEConfig, load_config, save_config, _dataclass_to_dict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Training orchestrator for the Region Proposal Transformer."""

    def __init__(self, config: TRIPTPEConfig):
        """Initialize the trainer.

        Args:
            config: Full TRIP-TPE configuration.
        """
        self.config = config
        self.tc = config.training
        self.mc = config.transformer

        # Resolve device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        print(f"Using device: {self.device}")

        # AMP is only enabled on CUDA. Numerically sensitive ops such as the
        # Beta KL path are still forced to float32 inside the loss.
        self.use_amp = bool(self.tc.fp16 and self.device.type == "cuda")
        print(f"Mixed precision: {'FP16 AMP' if self.use_amp else 'disabled'}")

        # Validate config consistency
        self._validate_config()

        # Build model (P1/P2 refinement flags propagated from config)
        self.model = RegionProposalTransformer(
            hp_dim=self.mc.hp_input_dim,
            obj_dim=self.mc.obj_input_dim,
            embed_dim=self.mc.embed_dim,
            num_heads=self.mc.num_heads,
            num_layers=self.mc.num_layers,
            ff_dim=self.mc.ff_dim,
            max_seq_len=self.mc.max_seq_len,
            dropout=self.mc.dropout,
            use_dimension_queries=getattr(self.mc, 'use_dimension_queries', False),
            dim_query_layers=getattr(self.mc, 'dim_query_layers', 1),
            n_beta_components=getattr(self.mc, 'n_beta_components', 1),
            meta_dim=getattr(self.mc, 'meta_dim', 0) if getattr(self.mc, 'use_meta_features', False) else 0,
        ).to(self.device)

        # Set meta-feature dropout from config (used during training only)
        meta_dropout = getattr(self.mc, 'meta_feature_dropout', 0.0)
        if meta_dropout > 0 and self.model.meta_projection is not None:
            self.model.meta_feature_dropout = meta_dropout
            print(f"Meta-feature dropout: {meta_dropout:.0%}")

        n_params = self.model.count_parameters()
        print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
        mem = self.model.memory_estimate_mb(self.tc.batch_size)
        print(f"Estimated VRAM: {mem['estimated_total_mb']:.0f} MB")

        # Loss function
        self.criterion = RegionProposalLoss(
            lambda_bound=1.0,
            lambda_cover=2.0,
            lambda_tight=0.3,
            lambda_kl=0.01,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Scheduler (cosine annealing with warmup)
        self.scheduler = None  # Set after dataloader is created

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

        # Weights & Biases integration
        self.use_wandb = WANDB_AVAILABLE and not getattr(config, '_disable_wandb', False)
        if self.use_wandb:
            try:
                wandb_kwargs = {
                    "project": getattr(config, 'wandb_project', 'trip-tpe'),
                    "name": getattr(config, 'wandb_run_name', None),
                    "config": _dataclass_to_dict(config),
                    "tags": ["training", f"layers-{self.mc.num_layers}", f"embed-{self.mc.embed_dim}"],
                    "save_code": True,
                }
                if getattr(config, 'wandb_entity', None):
                    wandb_kwargs["entity"] = config.wandb_entity
                
                wandb.init(**wandb_kwargs)
                wandb.watch(self.model, log="gradients", log_freq=100)
                print("W&B logging enabled.")
            except Exception as e:
                print(f"W&B init failed ({e}), continuing without logging.")
                self.use_wandb = False

    def _validate_config(self) -> None:
        """Validate configuration consistency before training.

        Raises:
            ValueError: If configuration values are inconsistent or invalid.
        """
        mc = self.mc
        tc = self.tc

        if mc.embed_dim % mc.num_heads != 0:
            raise ValueError(
                f"embed_dim ({mc.embed_dim}) must be divisible by "
                f"num_heads ({mc.num_heads})."
            )

        if mc.hp_input_dim < 1:
            raise ValueError(f"hp_input_dim must be >= 1, got {mc.hp_input_dim}.")

        if mc.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {mc.max_seq_len}.")

        if tc.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {tc.batch_size}.")

        if tc.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {tc.learning_rate}.")

        if tc.curriculum and tc.curriculum_start_len > tc.curriculum_end_len:
            raise ValueError(
                f"curriculum_start_len ({tc.curriculum_start_len}) must be <= "
                f"curriculum_end_len ({tc.curriculum_end_len})."
            )

        if tc.curriculum_end_len > mc.max_seq_len:
            import warnings
            warnings.warn(
                f"curriculum_end_len ({tc.curriculum_end_len}) exceeds "
                f"max_seq_len ({mc.max_seq_len}); clamping to max_seq_len.",
                UserWarning,
            )
            tc.curriculum_end_len = mc.max_seq_len

    def _build_scheduler(self, total_steps: int) -> None:
        """Build learning rate scheduler with linear warmup + cosine decay."""

        def lr_lambda(step: int) -> float:
            if step < self.tc.warmup_steps:
                return step / max(self.tc.warmup_steps, 1)
            progress = (step - self.tc.warmup_steps) / max(total_steps - self.tc.warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _curriculum_max_len(self, epoch: int) -> Optional[int]:
        """Compute the maximum sequence length for curriculum learning.

        Linearly ramps from curriculum_start_len to curriculum_end_len
        over curriculum_warmup_epochs, then stays at curriculum_end_len.

        Args:
            epoch: Current epoch (1-indexed).

        Returns:
            Maximum allowed sequence length, or None if curriculum is disabled.
        """
        tc = self.tc
        if not tc.curriculum:
            return None

        progress = min(epoch / max(tc.curriculum_warmup_epochs, 1), 1.0)
        max_len = int(
            tc.curriculum_start_len
            + progress * (tc.curriculum_end_len - tc.curriculum_start_len)
        )
        return max(max_len, tc.curriculum_start_len)

    def _apply_curriculum_truncation(
        self,
        input_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int,
    ) -> tuple:
        """Truncate sequences to the curriculum length.

        Args:
            input_seq: Shape (B, full_seq_len, feat_dim).
            attention_mask: Shape (B, full_seq_len).
            max_len: Maximum allowed sequence length.

        Returns:
            Truncated (input_seq, attention_mask).
        """
        current_len = input_seq.shape[1]
        if max_len >= current_len:
            return input_seq, attention_mask

        # Truncate and zero out the attention mask beyond max_len
        input_seq = input_seq[:, :max_len, :]
        attention_mask = attention_mask[:, :max_len]
        return input_seq, attention_mask

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of average loss components.
        """
        self.model.train()
        total_losses = {}
        n_batches = 0

        # Curriculum learning: compute max sequence length for this epoch
        curriculum_max_len = self._curriculum_max_len(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            # Move to device and sanitize any stray NaNs from datasets
            input_seq = torch.nan_to_num(batch["input_seq"]).to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_lower = torch.nan_to_num(batch["target_lower"]).to(self.device)
            target_upper = torch.nan_to_num(batch["target_upper"]).to(self.device)
            dim_mask = batch["dim_mask"].to(self.device)
            
            meta_features = batch.get("meta_features")
            if meta_features is not None:
                meta_features = torch.nan_to_num(meta_features, nan=0.5).to(self.device)

            # Apply curriculum truncation if active
            if curriculum_max_len is not None:
                input_seq, attention_mask = self._apply_curriculum_truncation(
                    input_seq, attention_mask, curriculum_max_len
                )

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                predictions = self.model(input_seq, attention_mask, meta_features=meta_features)
                losses = self.criterion(predictions, target_lower, target_upper, dim_mask)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            # Accumulate losses
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

            # W&B step-level logging
            if self.use_wandb and self.global_step % 10 == 0:
                log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                if curriculum_max_len is not None:
                    log_dict["train/curriculum_len"] = curriculum_max_len
                wandb.log(log_dict, step=self.global_step)

        # Average
        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of average validation metrics.
        """
        self.model.eval()
        total_losses = {}
        n_batches = 0

        # Additional metrics
        total_coverage = 0.0
        total_tightness = 0.0

        for batch in dataloader:
            input_seq = torch.nan_to_num(batch["input_seq"]).to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_lower = torch.nan_to_num(batch["target_lower"]).to(self.device)
            target_upper = torch.nan_to_num(batch["target_upper"]).to(self.device)
            dim_mask = batch["dim_mask"].to(self.device)
            meta_features = batch.get("meta_features")
            if meta_features is not None:
                meta_features = torch.nan_to_num(meta_features, nan=0.5).to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                predictions = self.model(input_seq, attention_mask, meta_features=meta_features)
                losses = self.criterion(predictions, target_lower, target_upper, dim_mask)

            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()

            # Coverage: fraction of dimensions where pred contains target
            pred_l = predictions["pred_lower"]
            pred_u = predictions["pred_upper"]
            mix_weights = predictions.get("mix_weights", None)

            # FIX: Collapse mixture components if using Mixture of Betas (K > 1)
            if mix_weights is not None and pred_l.dim() == 3:
                pred_l = (pred_l * mix_weights).sum(dim=-1)
                pred_u = (pred_u * mix_weights).sum(dim=-1)

            covers_lower = (pred_l <= target_lower + 0.01).float() * dim_mask
            covers_upper = (pred_u >= target_upper - 0.01).float() * dim_mask
            n_active = dim_mask.sum(dim=1).clamp(min=1.0)
            coverage = ((covers_lower * covers_upper).sum(dim=1) / n_active).mean()
            total_coverage += coverage.item()

            # Tightness: average predicted width
            pred_width = ((pred_u - pred_l) * dim_mask).sum(dim=1) / n_active
            total_tightness += pred_width.mean().item()

            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        avg["coverage"] = total_coverage / max(n_batches, 1)
        avg["avg_width"] = total_tightness / max(n_batches, 1)

        return avg

    def save_checkpoint(self, path: str, epoch: int, val_metrics: Dict) -> None:
        """Save a training checkpoint.

        Args:
            path: Output file path.
            epoch: Current epoch.
            val_metrics: Validation metrics at this checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_val_loss": self.best_val_loss,
            "model_config": {
                "hp_dim": self.mc.hp_input_dim,
                "obj_dim": self.mc.obj_input_dim,
                "embed_dim": self.mc.embed_dim,
                "num_heads": self.mc.num_heads,
                "num_layers": self.mc.num_layers,
                "ff_dim": self.mc.ff_dim,
                "max_seq_len": self.mc.max_seq_len,
                "dropout": self.mc.dropout,
                "use_dimension_queries": getattr(self.mc, 'use_dimension_queries', False),
                "dim_query_layers": getattr(self.mc, 'dim_query_layers', 1),
                "n_beta_components": getattr(self.mc, 'n_beta_components', 1),
                "meta_dim": getattr(self.mc, 'meta_dim', 0) if getattr(self.mc, 'use_meta_features', False) else 0,
            },
        }
        torch.save(checkpoint, path)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Best validation metrics achieved.
        """
        total_steps = self.tc.num_epochs * len(train_loader)
        self._build_scheduler(total_steps)

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_metrics = {}
        print(f"\nStarting training for {self.tc.num_epochs} epochs...")
        print(f"Total steps: {total_steps:,}")

        for epoch in range(1, self.tc.num_epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - t0

            # Log training metrics
            lr = self.optimizer.param_groups[0]["lr"]
            cur_len = self._curriculum_max_len(epoch)
            cur_suffix = f" | cur_len={cur_len}" if cur_len is not None else ""
            print(
                f"Epoch {epoch:3d} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"bound={train_metrics.get('bound_loss', 0):.4f} | "
                f"cover={train_metrics.get('coverage_loss', 0):.4f} | "
                f"lr={lr:.2e} | "
                f"time={epoch_time:.1f}s{cur_suffix}"
            )

            # Validate
            if epoch % self.tc.eval_every == 0 or epoch == self.tc.num_epochs:
                val_metrics = self.validate(val_loader)
                print(
                    f"  Val: loss={val_metrics['loss']:.4f} | "
                    f"coverage={val_metrics['coverage']:.3f} | "
                    f"avg_width={val_metrics['avg_width']:.3f}"
                )

                # Save best model
                # W&B epoch-level logging
                if self.use_wandb:
                    epoch_log = {f"val/{k}": v for k, v in val_metrics.items()}
                    epoch_log.update({f"train_epoch/{k}": v for k, v in train_metrics.items()})
                    epoch_log["epoch"] = epoch
                    epoch_log["epoch_time_s"] = epoch_time
                    wandb.log(epoch_log, step=self.global_step)

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    best_val_metrics = val_metrics
                    self.save_checkpoint(
                        str(ckpt_dir / "best_model.pt"), epoch, val_metrics
                    )
                    print(f"  New best model saved (val_loss={self.best_val_loss:.4f})")
                    if self.use_wandb:
                        wandb.run.summary["best_val_loss"] = self.best_val_loss
                        wandb.run.summary["best_epoch"] = epoch
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.tc.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.tc.patience})")
                    break

            # Periodic checkpoint
            if epoch % self.tc.save_every == 0:
                self.save_checkpoint(
                    str(ckpt_dir / f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    train_metrics,
                )

        # Save final model
        self.save_checkpoint(
            str(ckpt_dir / "final_model.pt"),
            epoch,
            train_metrics if not best_val_metrics else best_val_metrics,
        )
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        # Upload best model as W&B artifact
        if self.use_wandb:
            best_path = ckpt_dir / "best_model.pt"
            if best_path.exists():
                artifact = wandb.Artifact(
                    "trip-tpe-model", type="model",
                    description=f"Best model, val_loss={self.best_val_loss:.4f}",
                )
                artifact.add_file(str(best_path))
                wandb.log_artifact(artifact)
            wandb.finish()

        return best_val_metrics


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train TRIP-TPE Region Proposal Transformer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data only")
    parser.add_argument("--data-path", type=str, default=None, help="Path to .pt training data")
    parser.add_argument(
        "--mix-synthetic", type=float, default=0.1, metavar="FRAC",
        help="When using --data-path, mix this fraction of synthetic data "
             "for curriculum diversity (default: 0.1, 0 to disable). "
             "Real data should be the primary source; synthetic only augments.",
    )
    parser.add_argument("--n-trajectories", type=int, default=10000, help="Synthetic trajectories")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Load config
    overrides = {}
    if args.epochs:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.device:
        overrides["device"] = args.device
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    config = load_config(args.config, overrides if overrides else None)

    # W&B overrides
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config._disable_wandb = True

    # Build dataset
    if args.data_path:
        from trip_tpe.data.generate_trajectories import load_pairs
        pairs = load_pairs(args.data_path)
        primary_dataset = TrajectoryDataset(
            pairs,
            max_seq_len=config.transformer.max_seq_len,
            hp_dim=config.transformer.hp_input_dim,
        )
        print(f"Loaded {len(primary_dataset)} samples from {args.data_path}")

        # Optionally mix in synthetic data for curriculum diversity
        if args.mix_synthetic > 0:
            target_synth_items = max(
                1,
                int(len(primary_dataset) * args.mix_synthetic / (1.0 - args.mix_synthetic)),
            )
            synthetic_dataset = SyntheticTrajectoryDataset(
                n_trajectories=args.n_trajectories,
                hp_dim=config.transformer.hp_input_dim,
                max_seq_len=config.transformer.max_seq_len,
            )

            if len(synthetic_dataset) >= target_synth_items:
                synth_indices = torch.randperm(len(synthetic_dataset))[:target_synth_items].tolist()
                synthetic_dataset = Subset(synthetic_dataset, synth_indices)
            else:
                print(
                    f"Warning: requested {target_synth_items} synthetic items for "
                    f"mix_synthetic={args.mix_synthetic:.0%}, but only generated "
                    f"{len(synthetic_dataset)} from {args.n_trajectories} trajectories."
                )

            dataset = ConcatDataset([primary_dataset, synthetic_dataset])
            actual_synth_frac = len(synthetic_dataset) / max(len(dataset), 1)
            print(
                f"Mixed dataset: {len(primary_dataset)} surrogate + "
                f"{len(synthetic_dataset)} synthetic = {len(dataset)} total "
                f"({actual_synth_frac:.1%} synthetic, target={args.mix_synthetic:.0%})"
            )
        else:
            dataset = primary_dataset
    elif args.synthetic:
        dataset = SyntheticTrajectoryDataset(
            n_trajectories=args.n_trajectories,
            hp_dim=config.transformer.hp_input_dim,
            max_seq_len=config.transformer.max_seq_len,
        )
    else:
        # Default: abort with clear instructions. Training on synthetic-only
        # data produces a model that does not generalize to real HPO landscapes,
        # wasting cloud GPU hours. Require an explicit --synthetic flag to
        # override this safety check (useful for local testing only).
        print(
            "\n" + "=" * 70 + "\n"
            "ERROR: No --data-path specified and --synthetic flag not set.\n"
            "\n"
            "Training on synthetic data ONLY (quadratic/Rosenbrock/Ackley/Rastrigin)\n"
            "produces a weak model that does not generalize to real HPO landscapes.\n"
            "This would waste cloud GPU hours.\n"
            "\n"
            "For a production cloud training run, generate real data first:\n"
            "  trip-tpe-generate --mode real --output data/real_train.pt\n"
            "  (combines HPO-B + YAHPO Gym with aggressive augmentation)\n"
            "Then re-run with:\n"
            "  trip-tpe-train --data-path data/real_train.pt --mix-synthetic 0.1\n"
            "\n"
            "To intentionally train on synthetic data only (e.g. for local testing),\n"
            "pass the --synthetic flag explicitly.\n"
            + "=" * 70 + "\n"
        )
        sys.exit(1)

    # Train/val split
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Dataset: {n_total} samples (train={n_train}, val={n_val})")

    # Resolve collate_fn: check the underlying dataset class, not the Subset.
    # Both TrajectoryDataset and SyntheticTrajectoryDataset return identical
    # dict-of-tensor items, so always use TrajectoryDataset.collate_fn.
    _collate_fn = TrajectoryDataset.collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=min(config.training.num_workers, 2),
        pin_memory=config.training.pin_memory and torch.cuda.is_available(),
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=min(config.training.num_workers, 2),
        pin_memory=config.training.pin_memory and torch.cuda.is_available(),
        collate_fn=_collate_fn,
    )

    # Save config
    save_config(config, Path(config.checkpoint_dir) / "config.yaml")

    # Train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
