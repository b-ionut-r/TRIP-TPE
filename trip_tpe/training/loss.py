"""
Loss functions for training the Region Proposal Transformer.

Multi-objective loss combining:
    1. Region Bound Loss: MSE between predicted and ground-truth bounds
    2. Coverage Loss: Penalizes proposals that fail to contain the true optimum
    3. Tightness Loss: Rewards proposals that are as narrow as possible
    4. Beta KL Divergence: Regularizes the Beta distribution parameters
    5. Mode Accuracy Loss: Penalizes when the Beta distribution's mode is far
       from the target region center (NEW — critical for guided sampling mode)

The balance between coverage and tightness creates the core tension:
the model must learn to propose regions that are tight enough to help TPE
but broad enough to contain the global optimum.

For the guided sampling mode (recommended), the mode accuracy loss is the
most important term: it ensures the Beta distribution's peak is centered on
the actual promising region, so that samples drawn from the distribution
tend to land near the optimum.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionProposalLoss(nn.Module):
    """Combined loss for region proposal training.

    Loss = λ_bound * L_bound + λ_cover * L_cover + λ_tight * L_tight
         + λ_kl * L_kl + λ_mode * L_mode

    Where:
        L_bound: Smooth L1 loss between predicted and target bounds
        L_cover: Penalty for predicted regions that don't contain target bounds
        L_tight: Encourages tight (narrow) proposals
        L_kl: KL divergence regularization on Beta parameters
        L_mode: MSE between Beta distribution mode and target region center
    """

    def __init__(
        self,
        lambda_bound: float = 1.0,
        lambda_cover: float = 1.0,
        lambda_tight: float = 0.8,
        lambda_kl: float = 0.02,
        lambda_mode: float = 0.5,
    ):
        """Initialize the loss function.

        Args:
            lambda_bound: Weight for bound prediction loss.
            lambda_cover: Weight for coverage loss (higher = more conservative).
            lambda_tight: Weight for tightness reward (higher = tighter bounds).
            lambda_kl: Weight for Beta KL divergence regularization.
            lambda_mode: Weight for mode accuracy loss (higher = more focused
                         Beta peaks). Critical for guided sampling mode.
                         Set to 0.0 for legacy constrained-mode training.
        """
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_cover = lambda_cover
        self.lambda_tight = lambda_tight
        self.lambda_kl = lambda_kl
        self.lambda_mode = lambda_mode

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target_lower: torch.Tensor,
        target_upper: torch.Tensor,
        dim_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss.

        Supports both single-Beta (original) and mixture-of-Betas (P2).
        When predictions contain a component dimension (K > 1), losses are
        computed per-component and aggregated via mixing weights.

        Args:
            predictions: Output dict from RegionProposalTransformer.forward().
            target_lower: Ground truth lower bounds, shape (B, hp_dim).
            target_upper: Ground truth upper bounds, shape (B, hp_dim).
            dim_mask: Dimension mask, shape (B, hp_dim). 1=active, 0=padding.

        Returns:
            Dict with "loss" (total) and individual components.
        """
        alpha = predictions["alpha"]
        beta_param = predictions["beta"]
        mix_weights = predictions.get("mix_weights", None)

        # Detect P2 mixture mode: alpha has shape (B, hp_dim, K)
        is_mixture = alpha.dim() == 3 and alpha.shape[-1] > 1

        if is_mixture and mix_weights is not None:
            pred_lower = (predictions["pred_lower"] * mix_weights).sum(dim=-1)
            pred_upper = (predictions["pred_upper"] * mix_weights).sum(dim=-1)
            alpha_flat = (alpha * mix_weights).sum(dim=-1)
            beta_flat = (beta_param * mix_weights).sum(dim=-1)
        else:
            pred_lower = predictions["pred_lower"]
            pred_upper = predictions["pred_upper"]
            alpha_flat = alpha
            beta_flat = beta_param

        # Count active dimensions per sample for normalization
        n_active = dim_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # 1. Bound prediction loss (Smooth L1)
        lower_loss = F.smooth_l1_loss(pred_lower * dim_mask, target_lower * dim_mask, reduction="none")
        upper_loss = F.smooth_l1_loss(pred_upper * dim_mask, target_upper * dim_mask, reduction="none")
        bound_loss = ((lower_loss + upper_loss) * dim_mask).sum(dim=1) / n_active.squeeze(1)
        bound_loss = bound_loss.mean()

        # 2. Coverage loss: penalize when predicted region doesn't contain target region
        lower_violation = F.relu(pred_lower - target_lower) * dim_mask
        upper_violation = F.relu(target_upper - pred_upper) * dim_mask
        coverage_loss = (lower_violation + upper_violation).sum(dim=1) / n_active.squeeze(1)
        coverage_loss = coverage_loss.mean()

        # 3. Tightness loss: reward narrow proposals (minimize predicted width)
        pred_width = (pred_upper - pred_lower) * dim_mask
        tightness_loss = pred_width.sum(dim=1) / n_active.squeeze(1)
        tightness_loss = tightness_loss.mean()

        # 4. Beta KL divergence: regularize toward uniform Beta(1,1)
        prior_alpha = torch.ones_like(alpha_flat)
        prior_beta = torch.ones_like(beta_flat)
        kl = _beta_kl_divergence(alpha_flat, beta_flat, prior_alpha, prior_beta)
        kl_loss = (kl * dim_mask).sum(dim=1) / n_active.squeeze(1)
        kl_loss = kl_loss.mean()

        # 5. Mode accuracy loss: penalize Beta mode far from target center
        #    Beta mode = (α - 1) / (α + β - 2) when α, β > 1
        #    This is the single most important loss for guided sampling mode:
        #    it ensures that samples drawn from the Beta distribution land
        #    near the center of the actual promising region.
        mode_loss = torch.tensor(0.0, device=alpha_flat.device)
        if self.lambda_mode > 0:
            target_center = (target_lower + target_upper) / 2.0

            # Compute Beta mode, guarding against α,β ≤ 1 (where mode is
            # at the boundary, not interior). For α,β > 1 the mode formula
            # is well-defined. For other cases, use the mean α/(α+β) as a
            # smooth fallback.
            sum_ab = alpha_flat + beta_flat
            mode_denom = (sum_ab - 2.0).clamp(min=0.1)
            beta_mode_raw = (alpha_flat - 1.0).clamp(min=0.0) / mode_denom
            beta_mode_raw = beta_mode_raw.clamp(0.0, 1.0)

            # Fallback: use mean for dimensions where α or β ≤ 1
            beta_mean = alpha_flat / sum_ab.clamp(min=0.01)
            use_mode = ((alpha_flat > 1.0) & (beta_flat > 1.0)).float()
            beta_center = use_mode * beta_mode_raw + (1.0 - use_mode) * beta_mean

            mode_err = F.mse_loss(
                beta_center * dim_mask, target_center * dim_mask, reduction="none"
            )
            mode_loss = (mode_err * dim_mask).sum(dim=1) / n_active.squeeze(1)
            mode_loss = mode_loss.mean()

        # Combined loss
        total_loss = (
            self.lambda_bound * bound_loss
            + self.lambda_cover * coverage_loss
            + self.lambda_tight * tightness_loss
            + self.lambda_kl * kl_loss
            + self.lambda_mode * mode_loss
        )

        return {
            "loss": total_loss,
            "bound_loss": bound_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "tightness_loss": tightness_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mode_loss": mode_loss.detach(),
        }


def _beta_kl_divergence(
    alpha_q: torch.Tensor,
    beta_q: torch.Tensor,
    alpha_p: torch.Tensor,
    beta_p: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence KL(Beta(α_q, β_q) || Beta(α_p, β_p)).

    Uses the closed-form KL divergence for Beta distributions.

    Args:
        alpha_q, beta_q: Parameters of the approximate posterior.
        alpha_p, beta_p: Parameters of the prior.

    Returns:
        KL divergence per element, same shape as inputs.
    """
    # FIX: Force float32. lgamma and digamma are highly unstable in FP16
    # and will overflow to Infinity, causing the NaN losses.
    alpha_q, beta_q = alpha_q.float(), beta_q.float()
    alpha_p, beta_p = alpha_p.float(), beta_p.float()

    # Log Beta function: log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    log_beta_q = torch.lgamma(alpha_q) + torch.lgamma(beta_q) - torch.lgamma(alpha_q + beta_q)
    log_beta_p = torch.lgamma(alpha_p) + torch.lgamma(beta_p) - torch.lgamma(alpha_p + beta_p)

    kl = (
        log_beta_p - log_beta_q
        + (alpha_q - alpha_p) * torch.digamma(alpha_q)
        + (beta_q - beta_p) * torch.digamma(beta_q)
        + (alpha_p - alpha_q + beta_p - beta_q) * torch.digamma(alpha_q + beta_q)
    )

    return kl.clamp(min=0.0)  # Numerical safety
