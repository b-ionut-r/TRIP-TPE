"""
Loss functions for training the Region Proposal Transformer.

Multi-objective loss combining:
    1. Region Bound Loss: MSE between predicted and ground-truth bounds
    2. Coverage Loss: Penalizes proposals that fail to contain the true optimum
    3. Tightness Loss: Rewards proposals that are as narrow as possible
    4. Beta KL Divergence: Regularizes the Beta distribution parameters

The balance between coverage and tightness creates the core tension:
the model must learn to propose regions that are tight enough to help TPE
but broad enough to contain the global optimum.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionProposalLoss(nn.Module):
    """Combined loss for region proposal training.

    Loss = λ_bound * L_bound + λ_cover * L_cover + λ_tight * L_tight + λ_kl * L_kl

    Where:
        L_bound: Smooth L1 loss between predicted and target bounds
        L_cover: Penalty for predicted regions that don't contain target bounds
        L_tight: Encourages tight (narrow) proposals
        L_kl: KL divergence regularization on Beta parameters
    """

    def __init__(
        self,
        lambda_bound: float = 1.0,
        lambda_cover: float = 2.0,
        lambda_tight: float = 0.3,
        lambda_kl: float = 0.01,
    ):
        """Initialize the loss function.

        Args:
            lambda_bound: Weight for bound prediction loss.
            lambda_cover: Weight for coverage loss (higher = more conservative).
            lambda_tight: Weight for tightness reward (higher = tighter bounds).
            lambda_kl: Weight for Beta KL divergence regularization.
        """
        super().__init__()
        self.lambda_bound = lambda_bound
        self.lambda_cover = lambda_cover
        self.lambda_tight = lambda_tight
        self.lambda_kl = lambda_kl

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
            # P2: Collapse mixture to expected bounds using mixing weights
            # pred_lower/upper: (B, hp_dim, K), mix_weights: (B, hp_dim, K)
            pred_lower = (predictions["pred_lower"] * mix_weights).sum(dim=-1)
            pred_upper = (predictions["pred_upper"] * mix_weights).sum(dim=-1)
            # For KL: weighted average across components
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

        # Combined loss
        total_loss = (
            self.lambda_bound * bound_loss
            + self.lambda_cover * coverage_loss
            + self.lambda_tight * tightness_loss
            + self.lambda_kl * kl_loss
        )

        return {
            "loss": total_loss,
            "bound_loss": bound_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "tightness_loss": tightness_loss.detach(),
            "kl_loss": kl_loss.detach(),
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
