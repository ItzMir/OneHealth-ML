"""
OneHealth-Net: A Leakage-Audited Multi-Task Transformer for
Seven Chronic Disease Prediction.

Architecture:
1. Per-feature affine tokenizer (each feature has its own weight + bias).
2. Leaked features replaced with a learned [MASK] token BEFORE self-attention.
3. Shared transformer encoder (2 layers, 6 heads).
4. Per-disease learnable queries cross-attend over masked encoded tokens.
5. Multi-task heads (one per disease) + auxiliary head (metabolic count).
"""

import torch
import torch.nn as nn


class OneHealthNet(nn.Module):
    """
    Multi-task transformer for simultaneous prediction of 7 chronic diseases.

    Parameters
    ----------
    n_features : int
        Number of input features (after leakage masking, the full set is retained).
    n_diseases : int
        Number of disease labels to predict (7).
    leak_mask : np.ndarray of shape (n_diseases, n_features)
        Boolean mask. True indicates the feature is a leak for that disease
        and should be replaced with [MASK] before attention.
    d_model : int
        Transformer hidden dimension (default 96).
    n_heads : int
        Number of attention heads (default 6).
    n_layers : int
        Number of transformer encoder layers (default 2).
    dropout : float
        Dropout rate (default 0.15).

    Input
    -----
    x : torch.Tensor of shape (batch, n_features)

    Output
    ------
    logits : torch.Tensor of shape (batch, n_diseases)
    aux : torch.Tensor of shape (batch, 1)
    """

    def __init__(self, n_features, n_diseases, leak_mask,
                 d_model=96, n_heads=6, n_layers=2, dropout=0.15):
        super().__init__()
        self.n_features = n_features
        self.n_diseases = n_diseases

        # Leakage mask as a buffer (moves with model to device)
        self.register_buffer(
            "leak_mask",
            torch.tensor(leak_mask, dtype=torch.bool),
        )

        # Per-feature affine tokenizer
        self.tok_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.tok_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Per-disease learnable queries + cross-attention pooling
        self.disease_queries = nn.Parameter(torch.randn(n_diseases, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Disease heads
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(n_diseases)]
        )
        # Auxiliary head: predicts met_syndrome_count
        self.aux_head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, F = x.shape
        D = self.n_diseases

        # Tokenize: (B, F) -> (B, F, d_model)
        base = x.unsqueeze(-1) * self.tok_weight.unsqueeze(0) + self.tok_bias.unsqueeze(0)

        # Create per-disease masked views: (B, D, F, d_model)
        tokens = base.unsqueeze(1).expand(B, D, F, -1).clone()
        m = self.leak_mask.unsqueeze(0).expand(B, D, F)
        tokens[m] = self.mask_embed

        # Flatten for shared encoder: (B*D, F, d_model)
        tokens = tokens.reshape(B * D, F, -1)
        h = self.norm(self.encoder(tokens))

        # Per-disease query attention: (B*D, 1, d_model)
        q = self.disease_queries.unsqueeze(0).expand(B, D, -1).reshape(B * D, 1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        pooled = pooled.reshape(B, D, -1)

        # Disease logits: (B, D)
        logits = torch.stack(
            [self.heads[i](pooled[:, i]).squeeze(-1) for i in range(D)],
            dim=1,
        )

        # Auxiliary output
        aux = self.aux_head(pooled.mean(dim=1))

        return logits, aux