"""
Adaptive Fusion layer combining TA-BiLSTM temporal embeddings and GCN
structural embeddings (manuscript Section 4.6).
"""
import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    def __init__(self, temporal_dim: int, structural_dim: int, fused_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + structural_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.proj_temporal = nn.Linear(temporal_dim, fused_dim)
        self.proj_structural = nn.Linear(structural_dim, fused_dim)

    def forward(self, temporal_emb, structural_emb):
        concat = torch.cat([temporal_emb, structural_emb], dim=-1)
        gate_weights = self.gate(concat)
        fused = gate_weights * self.proj_temporal(temporal_emb) + \
                (1 - gate_weights) * self.proj_structural(structural_emb)
        return fused