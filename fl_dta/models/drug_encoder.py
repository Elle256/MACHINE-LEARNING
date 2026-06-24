# -*- coding: utf-8 -*-
"""
models/drug_encoder.py
----------------------
GNN encoder cho drug molecule.

Input  : PyG Batch (node features + edge_index)
Output : torch.Tensor  shape (batch_size, output_dim)

Kiến trúc: GCN → Global Mean Pooling → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class DrugGNNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder cho drug molecule.

    Parameters
    ----------
    num_features : int
        Số node features (mặc định 78 với featurizer của RDKit).
    hidden_dim : int
        Hidden dimension của GCN layers.
    output_dim : int
        Output embedding dimension (phải khớp với protein encoder).
    num_layers : int
        Số GCN layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_features: int = 168,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = num_features
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        # Projection head sau pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Batch
            Batch của drug graphs.

        Returns
        -------
        torch.Tensor  shape (batch_size, output_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling: (num_nodes, hidden) → (batch_size, hidden)
        x = global_mean_pool(x, batch)

        # Projection
        x = self.fc(x)
        return x
