"""
Region Proposal Transformer for TRIP-TPE.

A lightweight Transformer encoder (~10M parameters) that ingests partial HPO
trajectories and outputs per-dimension probability distributions defining
promising search space regions.

Architecture:
    Input: Sequence of (config, objective) pairs from partial optimization history
    Encoder: 4-layer Transformer with multi-head self-attention
    Output: Per-dimension Beta distribution parameters (α, β) defining region bounds

Design Rationale:
    - Beta distributions naturally model [0, 1]-bounded region proposals
    - The Transformer processes the optimization history as a sequence,
      leveraging attention to identify which early trials are most informative
    - Separate input projections for config features and objective values
      allow the model to learn distinct representations for "where" and "how good"
    - A [CLS]-style aggregation token pools information across the sequence
      before projecting to per-dimension region proposals

VRAM Budget (FP16, GTX 1650 Ti 4GB):
    - Static: ~120 MB (weights + gradients + optimizer)
    - Dynamic: ~1.5-2.5 GB (activations, batch_size=32, seq_len=200)
    - Total: ~2-3 GB → safely within 4 GB limit
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrialEmbedding(nn.Module):
    """Projects raw (config, objective) trial tokens into the embedding space.

    Each trial in the HPO sequence is represented as:
        [hp_1, hp_2, ..., hp_d, objective_value]

    This module applies separate linear projections to the hyperparameter
    configuration and the objective value, then combines them.
    """

    def __init__(
        self,
        hp_dim: int = 16,
        obj_dim: int = 1,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hp_proj = nn.Linear(hp_dim, embed_dim // 2)
        self.obj_proj = nn.Linear(obj_dim, embed_dim // 2)
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_seq: torch.Tensor,
        hp_dim: int = 16,
    ) -> torch.Tensor:
        """Embed a sequence of trial tokens.

        Args:
            input_seq: Shape (batch, seq_len, hp_dim + obj_dim).
            hp_dim: Number of hyperparameter dimensions.

        Returns:
            Embedded sequence, shape (batch, seq_len, embed_dim).
        """
        hp_features = input_seq[..., :hp_dim]
        obj_features = input_seq[..., hp_dim:]

        hp_emb = F.gelu(self.hp_proj(hp_features))
        obj_emb = F.gelu(self.obj_proj(obj_features))

        combined = torch.cat([hp_emb, obj_emb], dim=-1)
        combined = self.combine(combined)
        combined = self.norm(combined)
        return self.dropout(combined)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for optimization step positions.

    Unlike standard sinusoidal encodings, learned positions allow the model
    to assign different importance to early vs. late trials in the sequence,
    which is critical for understanding optimization dynamics.
    """

    def __init__(self, max_seq_len: int = 200, embed_dim: int = 256):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len + 1, embed_dim)
        # +1 for the [CLS] token at position 0
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate positional encodings.

        Args:
            seq_len: Current sequence length.
            device: Target device.

        Returns:
            Positional encodings, shape (1, seq_len, embed_dim).
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.position_embeddings(positions)


class DimensionCrossAttention(nn.Module):
    """P1 Refinement: DETR-style cross-attention dimension queries.

    Instead of compressing the full trajectory into a single [CLS] token,
    this module uses hp_dim learnable "dimension queries" that each
    cross-attend to the encoded trajectory, extracting dimension-specific
    information for region proposal.

    This removes the information bottleneck where a single 256-d vector
    must encode region-relevant information for all dimensions simultaneously.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hp_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hp_dim = hp_dim
        self.embed_dim = embed_dim

        # Learnable dimension queries — each specializes in one HP dimension
        self.dimension_queries = nn.Parameter(
            torch.randn(hp_dim, embed_dim) * 0.02
        )

        # Cross-attention layers: queries attend to encoded trajectory
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True,
                )
            )
            self.cross_norms.append(nn.LayerNorm(embed_dim))
            self.ff_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout),
            ))
            self.ff_norms.append(nn.LayerNorm(embed_dim))

    def forward(
        self,
        encoded_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend dimension queries to the encoded trajectory.

        Args:
            encoded_seq: Transformer encoder output, shape (B, S, D).
            key_padding_mask: Padding mask, shape (B, S). True = ignore.

        Returns:
            Per-dimension representations, shape (B, hp_dim, D).
        """
        batch_size = encoded_seq.shape[0]
        # Expand queries for the batch
        queries = self.dimension_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, hp_dim, D)

        for cross_attn, cross_norm, ff, ff_norm in zip(
            self.cross_attn_layers, self.cross_norms,
            self.ff_layers, self.ff_norms,
        ):
            # Pre-norm cross-attention
            q_normed = cross_norm(queries)
            attn_out, _ = cross_attn(
                query=q_normed,
                key=encoded_seq,
                value=encoded_seq,
                key_padding_mask=key_padding_mask,
            )
            queries = queries + attn_out

            # Pre-norm feed-forward
            queries = queries + ff(ff_norm(queries))

        return queries  # (B, hp_dim, D)


class RegionProposalHead(nn.Module):
    """Transforms the [CLS] token representation into per-dimension region proposals.

    Outputs Beta distribution parameters (α, β) for each hyperparameter dimension.
    The Beta distribution naturally constrains proposals to [0, 1] and can express:
        - Tight, confident regions (high α and β near the mode)
        - Broad, uncertain regions (α ≈ 1, β ≈ 1, near uniform)
        - Skewed regions (asymmetric α, β)
        - U-shaped / boundary-concentrated regions (α < 1 or β < 1)

    Region bounds are derived from the Beta distribution's quantiles at
    a configurable confidence level.

    P2 Refinement: Supports K-component Mixture-of-Betas when n_components > 1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hp_dim: int = 16,
        hidden_dim: int = 512,
        n_components: int = 1,
        per_dim_input: bool = False,
    ):
        """Initialize the region proposal head.

        Args:
            embed_dim: Input embedding dimension.
            hp_dim: Number of hyperparameter dimensions.
            hidden_dim: Hidden layer dimension.
            n_components: Number of Beta mixture components (P2).
                          1 = original single-Beta behavior.
            per_dim_input: If True, input is (B, hp_dim, D) from DETR queries
                           instead of (B, D) from [CLS]. (P1 support)
        """
        super().__init__()
        self.hp_dim = hp_dim
        self.n_components = n_components
        self.per_dim_input = per_dim_input

        if per_dim_input:
            # P1: Each dimension has its own MLP (shared weights via a single
            # MLP applied per-dim for efficiency)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            # Per-dim output: 1 alpha, 1 beta, 1 lower, 1 upper per component
            # + mixing weights if n_components > 1
            self.alpha_head = nn.Linear(hidden_dim, n_components)
            self.beta_head = nn.Linear(hidden_dim, n_components)
            self.lower_head = nn.Linear(hidden_dim, n_components)
            self.upper_head = nn.Linear(hidden_dim, n_components)
            if n_components > 1:
                self.mix_head = nn.Linear(hidden_dim, n_components)
        else:
            # Original: single [CLS] → all dimensions
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            # Output n_components * hp_dim for each parameter
            self.alpha_head = nn.Linear(hidden_dim, hp_dim * n_components)
            self.beta_head = nn.Linear(hidden_dim, hp_dim * n_components)
            self.lower_head = nn.Linear(hidden_dim, hp_dim * n_components)
            self.upper_head = nn.Linear(hidden_dim, hp_dim * n_components)
            if n_components > 1:
                self.mix_head = nn.Linear(hidden_dim, hp_dim * n_components)

    def forward(
        self, cls_token: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict region proposals.

        Args:
            cls_token: Shape (batch, embed_dim) for [CLS] mode,
                       or (batch, hp_dim, embed_dim) for per-dim DETR mode.

        Returns:
            Dictionary with:
                - "alpha": shape (batch, hp_dim) or (batch, hp_dim, K)
                - "beta": same shape as alpha
                - "pred_lower", "pred_upper": same shape as alpha
                - "mix_weights": shape (batch, hp_dim, K) if K > 1
        """
        K = self.n_components

        if self.per_dim_input:
            # P1 path: cls_token is (B, hp_dim, D)
            h = self.mlp(cls_token)  # (B, hp_dim, hidden)
            raw_alpha = self.alpha_head(h)  # (B, hp_dim, K)
            raw_beta = self.beta_head(h)    # (B, hp_dim, K)
            raw_lower = self.lower_head(h)  # (B, hp_dim, K)
            raw_upper = self.upper_head(h)  # (B, hp_dim, K)
        else:
            # Original [CLS] path: cls_token is (B, D)
            h = self.mlp(cls_token)  # (B, hidden)
            B = h.shape[0]
            raw_alpha = self.alpha_head(h).view(B, self.hp_dim, K)
            raw_beta = self.beta_head(h).view(B, self.hp_dim, K)
            raw_lower = self.lower_head(h).view(B, self.hp_dim, K)
            raw_upper = self.upper_head(h).view(B, self.hp_dim, K)

        # Softplus ensures α, β > 0 with epsilon for numerical safety.
        alpha = F.softplus(raw_alpha) + 0.01
        beta_param = F.softplus(raw_beta) + 0.01

        # Direct bound predictions (sigmoid → [0, 1])
        pred_lower = torch.sigmoid(raw_lower)
        pred_upper = torch.sigmoid(raw_upper)

        # Ensure pred_lower < pred_upper per component
        lower_clamped = torch.min(pred_lower, pred_upper)
        upper_clamped = torch.max(pred_lower, pred_upper)

        result: Dict[str, torch.Tensor] = {}

        if K == 1:
            # Squeeze out the component dimension for backward compatibility
            result["alpha"] = alpha.squeeze(-1)         # (B, hp_dim)
            result["beta"] = beta_param.squeeze(-1)
            result["pred_lower"] = lower_clamped.squeeze(-1)
            result["pred_upper"] = upper_clamped.squeeze(-1)
        else:
            # P2: keep component dimension
            result["alpha"] = alpha           # (B, hp_dim, K)
            result["beta"] = beta_param
            result["pred_lower"] = lower_clamped
            result["pred_upper"] = upper_clamped
            # Mixing weights via softmax.
            # h is already the MLP output: (B, hidden) for [CLS] mode,
            # or (B, hp_dim, hidden) for per-dim DETR mode. Reuse directly.
            mix_logits = self.mix_head(h)
            # When per_dim_input=False, mix_logits is (B, hp_dim*K) → reshape.
            # When per_dim_input=True, mix_logits is (B, hp_dim, K) already.
            if not self.per_dim_input:
                mix_logits = mix_logits.view(h.shape[0], self.hp_dim, K)
            result["mix_weights"] = F.softmax(mix_logits, dim=-1)  # (B, hp_dim, K)

        return result


class MetaFeatureProjection(nn.Module):
    """Projects dataset-level meta-features into the Transformer embedding space.

    Used for cold-start conditioning: when the Transformer has seen zero or few
    trials from the current task, the meta-features provide a prior signal about
    the dataset characteristics (size, dimensionality, class balance, etc.) that
    helps the model retrieve relevant patterns from training.

    The projected meta-features are ADDED to the [CLS] token embedding before
    the Transformer encoder processes the sequence. This is architecturally
    cleaner than prepending a [META] token because it:
        - Preserves sequence length and positional encoding semantics
        - Avoids attention mask complexity
        - Directly biases the global summary token, which is the bottleneck
          through which all region proposals are generated
    """

    def __init__(
        self,
        meta_dim: int = 7,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize the meta-feature projection.

        Args:
            meta_dim: Number of dataset-level meta-features (default 7 = Tier 1).
            embed_dim: Target embedding dimension (must match Transformer embed_dim).
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(meta_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        """Project meta-features into the embedding space.

        Args:
            meta_features: Shape (batch, meta_dim).

        Returns:
            Projected meta-features, shape (batch, embed_dim).
        """
        return self.projection(meta_features)


class RegionProposalTransformer(nn.Module):
    """Complete Region Proposal Transformer for TRIP-TPE.

    Architecture overview:
        1. TrialEmbedding: Projects (config, objective) → embed_dim
        2. Prepend [CLS] token for sequence aggregation
        2b. (Optional) Add projected dataset meta-features to [CLS] token
        3. Add learned positional encodings
        4. N-layer Transformer encoder with self-attention
        5. RegionProposalHead: [CLS] → per-dimension Beta distributions + bounds

    Refinements (backward-compatible, off by default):
        P1: use_dimension_queries=True replaces [CLS] with DETR-style
            per-dimension cross-attention queries.
        P2: n_beta_components > 1 enables Mixture-of-Betas output.
        Meta-features: meta_dim > 0 enables cold-start conditioning via
            dataset-level features added to the [CLS] token.

    Parameters (~10M with default config):
        - TrialEmbedding: ~67K
        - Positional: ~51K
        - Transformer (4 layers): ~8.5M
        - RegionProposalHead: ~400K
        - MetaFeatureProjection: ~2K (meta_dim=7)
        - [CLS] token: 256
        Total: ~9M parameters
    """

    def __init__(
        self,
        hp_dim: int = 16,
        obj_dim: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        use_dimension_queries: bool = False,
        dim_query_layers: int = 1,
        n_beta_components: int = 1,
        meta_dim: int = 0,
    ):
        """Initialize the Region Proposal Transformer.

        Args:
            hp_dim: Max hyperparameter dimensions.
            obj_dim: Number of objective values per trial.
            embed_dim: Transformer hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            ff_dim: Feed-forward hidden dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout rate.
            use_dimension_queries: P1 — use DETR-style dimension queries
                instead of [CLS] token.
            dim_query_layers: Number of cross-attention layers for P1.
            n_beta_components: P2 — number of Beta mixture components.
            meta_dim: Number of dataset-level meta-features for cold-start
                conditioning. 0 = disabled (backward-compatible default).
                7 = Tier 1 features (log_n_samples, log_n_features, n_classes,
                dimensionality_ratio, imbalance_ratio, frac_categorical,
                frac_missing).
        """
        super().__init__()
        self.hp_dim = hp_dim
        self.obj_dim = obj_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_dimension_queries = use_dimension_queries
        self.n_beta_components = n_beta_components
        self.meta_dim = meta_dim

        # Trial embedding
        self.trial_embedding = TrialEmbedding(
            hp_dim=hp_dim,
            obj_dim=obj_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # [CLS] token for sequence aggregation (kept even with P1 for
        # backward compat and as a global summary token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Meta-feature projection for cold-start conditioning
        # meta_feature_dropout: probability of zeroing the entire meta-feature
        # vector during training. Set by Trainer from config. Analogous to
        # classifier-free guidance dropout — forces the model to also work
        # without meta-features, preventing over-reliance on dataset identity.
        self.meta_projection: Optional[MetaFeatureProjection] = None
        self.meta_feature_dropout: float = 0.0
        if meta_dim > 0:
            self.meta_projection = MetaFeatureProjection(
                meta_dim=meta_dim,
                embed_dim=embed_dim,
                dropout=dropout,
            )

        # Positional encoding (+1 for CLS)
        self.pos_encoding = LearnedPositionalEncoding(
            max_seq_len=max_seq_len + 1,
            embed_dim=embed_dim,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # P1: Optional DETR-style cross-attention dimension queries
        self.dim_cross_attn: Optional[DimensionCrossAttention] = None
        if use_dimension_queries:
            self.dim_cross_attn = DimensionCrossAttention(
                embed_dim=embed_dim,
                hp_dim=hp_dim,
                num_heads=num_heads,
                num_layers=dim_query_layers,
                dropout=dropout,
            )

        # Region proposal head
        self.region_head = RegionProposalHead(
            embed_dim=embed_dim,
            hp_dim=hp_dim,
            hidden_dim=ff_dim // 2,
            n_components=n_beta_components,
            per_dim_input=use_dimension_queries,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier/Kaiming initialization for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        meta_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: trajectory → region proposals.

        Args:
            input_seq: Sequence of (config || objective) tokens.
                       Shape: (batch, seq_len, hp_dim + obj_dim).
            attention_mask: Binary mask, shape (batch, seq_len).
                           1 = valid token, 0 = padding.
            meta_features: Optional dataset-level meta-features for cold-start
                          conditioning. Shape: (batch, meta_dim).
                          When provided and meta_dim > 0, the projected features
                          are added to the [CLS] token embedding, biasing the
                          global summary toward dataset-appropriate regions.

        Returns:
            Dictionary with region proposal outputs from RegionProposalHead,
            plus "cls_embedding" for potential downstream use.
        """
        batch_size, seq_len, _ = input_seq.shape

        # 1. Embed trial tokens
        embedded = self.trial_embedding(input_seq, hp_dim=self.hp_dim)

        # 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 2b. Condition [CLS] on dataset meta-features (cold-start signal)
        if self.meta_projection is not None and meta_features is not None:
            # Meta-feature dropout: during training, randomly zero out the
            # entire meta-feature vector for some samples. This prevents
            # over-reliance on dataset identity and forces the model to also
            # learn from trajectory patterns alone (analogous to classifier-
            # free guidance dropout in conditional generation).
            if self.training and self.meta_feature_dropout > 0:
                drop_mask = torch.rand(batch_size, 1, device=meta_features.device)
                meta_features = meta_features * (drop_mask > self.meta_feature_dropout).float()
            meta_emb = self.meta_projection(meta_features)  # (B, D)
            cls_tokens = cls_tokens + meta_emb.unsqueeze(1)  # (B, 1, D)

        embedded = torch.cat([cls_tokens, embedded], dim=1)  # (B, 1+S, D)

        # 3. Add positional encoding
        embedded = embedded + self.pos_encoding(seq_len + 1, embedded.device)

        # 4. Create attention mask for Transformer
        # PyTorch TransformerEncoder uses src_key_padding_mask where True = ignore
        if attention_mask is not None:
            # Add CLS position (always attend to CLS)
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 1+S)
            padding_mask = (full_mask == 0)  # True = pad position
        else:
            padding_mask = None

        # 5. Transformer encoder
        encoded = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # 6. Extract [CLS] token (always available as global summary)
        cls_output = encoded[:, 0, :]  # (B, D)

        # 7. Region proposal — P1 branch vs. original [CLS] branch
        if self.use_dimension_queries and self.dim_cross_attn is not None:
            # P1: Per-dimension queries cross-attend to the full encoded
            # sequence (excluding [CLS] to avoid information shortcut)
            dim_representations = self.dim_cross_attn(
                encoded_seq=encoded[:, 1:, :],  # skip [CLS]
                key_padding_mask=padding_mask[:, 1:] if padding_mask is not None else None,
            )  # (B, hp_dim, D)
            proposals = self.region_head(dim_representations)
        else:
            # Original: [CLS] → all dimensions
            proposals = self.region_head(cls_output)

        proposals["cls_embedding"] = cls_output

        return proposals

    def predict_region(
        self,
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        confidence_level: float = 0.9,
        dim_mask: Optional[torch.Tensor] = None,
        beta_blend_weight: float = 0.5,
        meta_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict region bounds from a partial trajectory.

        Convenience method that runs the forward pass and extracts
        concrete [lower, upper] bounds per dimension.

        Uses a dual strategy:
            1. Beta distribution quantiles at the given confidence level
            2. Direct bound predictions from the supervised head
            3. Weighted average of both (ensembling for robustness)

        Args:
            input_seq: Shape (batch, seq_len, hp_dim + obj_dim).
            attention_mask: Shape (batch, seq_len).
            confidence_level: Confidence level for Beta quantile bounds.
            dim_mask: Shape (batch, hp_dim) — which dims are active.
            beta_blend_weight: Weight for Beta quantile bounds in ensemble
                               (0.0 = only direct bounds, 1.0 = only Beta).
            meta_features: Optional dataset-level meta-features, shape
                          (batch, meta_dim). Passed through to forward().

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (batch, hp_dim).
            Values are in [0, 1] (normalized search space).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_seq, attention_mask, meta_features=meta_features)

        alpha = outputs["alpha"]
        beta_param = outputs["beta"]
        pred_lower = outputs["pred_lower"]
        pred_upper = outputs["pred_upper"]
        mix_weights = outputs.get("mix_weights", None)  # (B, hp_dim, K) or None

        # --- P2: Mixture-of-Betas quantile extraction ---
        if self.n_beta_components > 1 and mix_weights is not None:
            # For each dimension, compute bounds from the dominant component
            # (highest mixing weight). This is conservative; a more
            # sophisticated approach would take the union of all component
            # high-density regions.
            dominant_k = mix_weights.argmax(dim=-1)  # (B, hp_dim)
            B, D, K = alpha.shape
            batch_idx = torch.arange(B, device=alpha.device).unsqueeze(1).expand(B, D)
            dim_idx = torch.arange(D, device=alpha.device).unsqueeze(0).expand(B, D)
            alpha_sel = alpha[batch_idx, dim_idx, dominant_k]       # (B, hp_dim)
            beta_sel = beta_param[batch_idx, dim_idx, dominant_k]
            pred_lower_sel = pred_lower[batch_idx, dim_idx, dominant_k]
            pred_upper_sel = pred_upper[batch_idx, dim_idx, dominant_k]
        else:
            # Original single-component path (backward compatible)
            alpha_sel = alpha
            beta_sel = beta_param
            pred_lower_sel = pred_lower
            pred_upper_sel = pred_upper

        # Beta distribution quantile bounds
        tail = (1.0 - confidence_level) / 2.0
        try:
            from scipy import stats as sp_stats
            alpha_np = alpha_sel.cpu().numpy()
            beta_np = beta_sel.cpu().numpy()
            beta_lower = sp_stats.beta.ppf(tail, alpha_np, beta_np)
            beta_upper = sp_stats.beta.ppf(1.0 - tail, alpha_np, beta_np)
            beta_lower = torch.from_numpy(beta_lower).to(input_seq.device).float()
            beta_upper = torch.from_numpy(beta_upper).to(input_seq.device).float()
        except Exception as e:
            # Fallback: use mean ± scaled std of Beta distribution
            import warnings
            warnings.warn(
                f"scipy.stats.beta.ppf failed ({e}); using moment-based fallback "
                "for Beta quantile estimation. Results may be less accurate.",
                RuntimeWarning,
            )
            mean = alpha_sel / (alpha_sel + beta_sel)
            var = (alpha_sel * beta_sel) / ((alpha_sel + beta_sel)**2 * (alpha_sel + beta_sel + 1))
            std = torch.sqrt(var + 1e-8)
            z = 1.645 if confidence_level >= 0.9 else 1.96
            beta_lower = torch.clamp(mean - z * std, 0.0, 1.0)
            beta_upper = torch.clamp(mean + z * std, 0.0, 1.0)

        # Ensemble: weighted average of Beta quantiles and direct predictions
        w = beta_blend_weight
        lower = w * beta_lower + (1 - w) * pred_lower_sel
        upper = w * beta_upper + (1 - w) * pred_upper_sel

        # Ensure lower < upper with minimum width
        lower = torch.clamp(lower, 0.0, 1.0)
        upper = torch.clamp(upper, 0.0, 1.0)
        min_width = 0.05
        too_narrow = (upper - lower) < min_width
        midpoint = (upper + lower) / 2.0
        lower = torch.where(too_narrow, torch.clamp(midpoint - min_width / 2, min=0.0), lower)
        upper = torch.where(too_narrow, torch.clamp(midpoint + min_width / 2, max=1.0), upper)

        # Apply dimension mask: inactive dims get full [0, 1] range
        if dim_mask is not None:
            lower = lower * dim_mask
            upper = upper * dim_mask + (1.0 - dim_mask)

        return lower, upper

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def memory_estimate_mb(self, batch_size: int = 32, seq_len: int = 200) -> Dict[str, float]:
        """Estimate VRAM usage in MB.

        Args:
            batch_size: Training batch size.
            seq_len: Sequence length.

        Returns:
            Dict with memory breakdown.
        """
        n_params = self.count_parameters()
        weights_mb = n_params * 2 / 1e6  # FP16
        grads_mb = n_params * 2 / 1e6
        optimizer_mb = n_params * 8 / 1e6  # AdamW FP32 states
        static_mb = weights_mb + grads_mb + optimizer_mb

        # Activation estimate (rough): ~4 bytes per activation element
        act_elements = batch_size * (seq_len + 1) * self.embed_dim * self.transformer.num_layers * 4
        # Add cross-attention activations if P1 is enabled
        if self.dim_cross_attn is not None:
            act_elements += batch_size * self.hp_dim * self.embed_dim * 4
        # Meta-feature projection is negligible (~2K params)
        activation_mb = act_elements * 2 / 1e6  # FP16

        return {
            "weights_mb": weights_mb,
            "gradients_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "static_total_mb": static_mb,
            "activation_mb": activation_mb,
            "estimated_total_mb": static_mb + activation_mb,
        }
