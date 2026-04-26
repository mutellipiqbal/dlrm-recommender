"""
DLRM — Deep Learning Recommendation Model  (src/model.py)
===========================================================
Meta's DLRM architecture for the Re-Ranking stage.
Paper: "Deep Learning Recommendation Model for Personalization and
        Recommendation Systems" (Naumov et al., 2019)

Used by Meta, Twitter, Alibaba for fine-grained ranking of candidates
produced by a retrieval stage (e.g., Two-Tower model).

Architecture:
    Dense features ─► Bottom MLP ─────────────────────────────┐
                                                               ├─► Feature Interaction ─► Top MLP ─► sigmoid
    Sparse features ─► Embedding tables ─► [emb_1, emb_2, …] ┘
                        (user_id, item_id, category, location, …)

Inputs:
    - Dense  (continuous): user_age, item_price, days_since_signup, …
    - Sparse (categorical): user_id, item_id, country_id, category_id, …

Output: P(click) / P(purchase) — binary prediction
Loss  : BCEWithLogitsLoss
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Sub-modules ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Linear → BatchNorm → ReLU → Dropout stack."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.1,
        use_bn: bool = True,
        final_activation: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or final_activation:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseEmbeddings(nn.Module):
    """
    One nn.Embedding table per sparse feature.
    All tables share the same embedding_dim for the interaction layer.
    """

    def __init__(self, vocab_sizes: list[int], embedding_dim: int):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.Embedding(v + 1, embedding_dim, padding_idx=0)   # +1 for OOV / padding
            for v in vocab_sizes
        ])
        for t in self.tables:
            nn.init.xavier_uniform_(t.weight)

    def forward(self, sparse_inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            sparse_inputs: (B, num_sparse_fields) — integer IDs
        Returns:
            list of (B, embedding_dim) tensors, one per field
        """
        return [self.tables[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]


class FeatureInteraction(nn.Module):
    """
    Dot-product interaction between all pairs of (dense_proj, sparse_emb_1, …, sparse_emb_N).
    Returns upper-triangle of the interaction matrix, concatenated with the dense projection.
    """

    def forward(
        self,
        dense_proj: torch.Tensor,           # (B, D)
        sparse_embs: list[torch.Tensor],    # each (B, D)
    ) -> torch.Tensor:
        # Stack all vectors: (B, 1+N, D)
        all_vecs = torch.stack([dense_proj] + sparse_embs, dim=1)

        # Pairwise dot products: (B, 1+N, 1+N)
        interactions = torch.bmm(all_vecs, all_vecs.transpose(1, 2))

        # Upper triangle (excluding diagonal) → (B, T) where T = n*(n-1)/2
        n = all_vecs.shape[1]
        rows, cols = torch.triu_indices(n, n, offset=1)
        interact_flat = interactions[:, rows, cols]   # (B, T)

        # Concatenate dense projection + interaction features
        return torch.cat([dense_proj, interact_flat], dim=1)   # (B, D + T)


# ─── DLRM ─────────────────────────────────────────────────────────────────────

class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model (Naumov et al., 2019).

    Args:
        num_dense_features   : number of continuous input features
        vocab_sizes          : list of vocab sizes for each sparse (categorical) feature
        embedding_dim        : shared embedding dimension for all sparse features
        bottom_mlp_dims      : hidden dims for the Bottom MLP (dense features)
        top_mlp_dims         : hidden dims for the Top MLP (after feature interaction)
        dropout              : dropout rate in MLPs
    """

    def __init__(
        self,
        num_dense_features: int,
        vocab_sizes: list[int],
        embedding_dim: int = 16,
        bottom_mlp_dims: list[int] | None = None,
        top_mlp_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        bottom_mlp_dims = bottom_mlp_dims or [64, 16]
        top_mlp_dims    = top_mlp_dims    or [64, 32]

        # Bottom MLP: projects dense features to embedding_dim
        self.bottom_mlp = MLP(
            in_dim=num_dense_features,
            hidden_dims=bottom_mlp_dims[:-1],
            out_dim=embedding_dim,       # must match embedding_dim for interaction
            dropout=dropout,
            use_bn=True,
            final_activation=True,       # ReLU before interaction
        )

        # Sparse embedding tables
        self.sparse_embeddings = SparseEmbeddings(vocab_sizes, embedding_dim)
        self.num_sparse = len(vocab_sizes)

        # Feature interaction layer
        self.interaction = FeatureInteraction()

        # Compute top MLP input dim:
        # dense_proj (D) + upper-triangle pairs of (1+N, 1+N) matrix
        n = 1 + self.num_sparse
        num_interactions = n * (n - 1) // 2
        top_in_dim = embedding_dim + num_interactions

        # Top MLP: interaction → binary logit
        self.top_mlp = MLP(
            in_dim=top_in_dim,
            hidden_dims=top_mlp_dims,
            out_dim=1,
            dropout=dropout,
            use_bn=False,               # no BN on final layers (common in practice)
            final_activation=False,
        )

    def forward(
        self,
        dense_inputs: torch.Tensor,    # (B, num_dense_features)
        sparse_inputs: torch.Tensor,   # (B, num_sparse_fields)
    ) -> torch.Tensor:
        # 1. Dense path
        dense_proj = self.bottom_mlp(dense_inputs)   # (B, D)

        # 2. Sparse path
        sparse_embs = self.sparse_embeddings(sparse_inputs)   # list of (B, D)

        # 3. Feature interaction
        x = self.interaction(dense_proj, sparse_embs)         # (B, D + T)

        # 4. Top MLP → logit
        return self.top_mlp(x).squeeze(-1)                    # (B,)
