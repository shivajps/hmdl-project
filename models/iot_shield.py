"""
Full IoT-SHIELD model: CLF-GWO-selected features -> TA-BiLSTM -> GCN ->
Adaptive Fusion -> classification head (manuscript Section 4, Figure 1).
"""
import torch
import torch.nn as nn

from ta_bilstm import TABiLSTM
from gcn import GCNEncoder
from adaptive_fusion import AdaptiveFusion


class IoTShield(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 lstm_hidden=128, lstm_layers=2,
                 gcn_hidden=64, gcn_layers=2,
                 fused_dim=128, dropout=0.3):
        super().__init__()
        self.temporal_encoder = TABiLSTM(input_dim, lstm_hidden, lstm_layers, dropout)
        self.structural_encoder = GCNEncoder(input_dim, gcn_hidden, gcn_layers, dropout)
        self.fusion = AdaptiveFusion(self.temporal_encoder.output_dim,
                                     self.structural_encoder.output_dim, fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, seq_x, graph_x, edge_index, graph_batch_index=None):
        temporal_emb, attn_weights = self.temporal_encoder(seq_x)
        structural_emb = self.structural_encoder(graph_x, edge_index)

        if graph_batch_index is not None:
            structural_emb = _pool_by_graph(structural_emb, graph_batch_index)

        fused = self.fusion(temporal_emb, structural_emb)
        logits = self.classifier(fused)
        return logits, attn_weights


def _pool_by_graph(node_embeddings, batch_index):
    """Mean-pool node embeddings per graph in the batch."""
    num_graphs = batch_index.max().item() + 1
    dim = node_embeddings.size(-1)
    pooled = torch.zeros(num_graphs, dim, device=node_embeddings.device)
    counts = torch.zeros(num_graphs, 1, device=node_embeddings.device)
    pooled.index_add_(0, batch_index, node_embeddings)
    counts.index_add_(0, batch_index, torch.ones(node_embeddings.size(0), 1, device=node_embeddings.device))
    return pooled / counts.clamp(min=1)