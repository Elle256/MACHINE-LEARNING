# -*- coding: utf-8 -*-
"""
models/protein_encoder.py
--------------------------
CNN encoder cho protein sequence (từ code gốc, đã refactor và fix bug).

Input  : torch.LongTensor  shape (batch_size, max_seq_len)
Output : torch.Tensor      shape (batch_size, output_dim)

Fix so với code gốc:
  - Thêm super().__init__() bị thiếu trong EmbeddingProtein
  - Sửa target.target → target (data không phải Batch ở đây)
  - Tính FLATTEN_SHAPE động thay vì hardcode
"""

import torch
import torch.nn as nn


class ProteinCNNEncoder(nn.Module):
    """
    CNN encoder cho protein sequence.

    Parameters
    ----------
    num_features_protein : int
        Số loại amino acid trong vocabulary (mặc định 25: 20 + unknown + pad).
    embed_dim : int
        Embedding dimension cho mỗi amino acid.
    n_filters : int
        Số filters của Conv1d layer.
    kernel_size : int
        Kernel size của Conv1d.
    max_seq_len : int
        Độ dài tối đa của sequence (dùng để tính flatten shape).
    output_dim : int
        Output embedding dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_features_protein: int = 25,
        embed_dim: int = 25,
        n_filters: int = 32,
        kernel_size: int = 8,
        max_seq_len: int = 1000,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()  # Fix: code gốc thiếu dòng này

        self.num_features_protein = num_features_protein
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.output_dim = output_dim

        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        # +1 vì index 0 là padding
        self.embedding = nn.Embedding(
            num_embeddings=num_features_protein + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # Conv1d input: (batch, seq_len, embed_dim) cần permute → (batch, embed_dim, seq_len)
        # Sau conv: (batch, n_filters, seq_len - kernel_size + 1)
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=n_filters,
            kernel_size=kernel_size,
        )
        self.bn1 = nn.BatchNorm1d(n_filters)

        self.conv2 = nn.Conv1d(
            in_channels=n_filters,
            out_channels=n_filters * 2,
            kernel_size=kernel_size,
        )
        self.bn2 = nn.BatchNorm1d(n_filters * 2)

        self.conv3 = nn.Conv1d(
            in_channels=n_filters * 2,
            out_channels=n_filters * 3,
            kernel_size=kernel_size,
        )
        self.bn3 = nn.BatchNorm1d(n_filters * 3)

        # Tính flatten shape động
        flatten_dim = self._compute_flatten_dim(max_seq_len, embed_dim, n_filters, kernel_size)

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _compute_flatten_dim(
        seq_len: int, embed_dim: int, n_filters: int, kernel_size: int
    ) -> int:
        """Tính output size sau 3 Conv1d layers."""
        # Sau conv1: seq_len - k + 1
        # Sau conv2: seq_len - 2*(k-1) - 1
        # Sau conv3: seq_len - 3*(k-1) - 2
        # Cụ thể với default (1000, 25, 32, 8):
        #   1000 - 7 = 993 → 993 - 7 = 986 → 986 - 7 = 979
        #   → flatten = 979 * (32 * 3) = 979 * 96 = 93984
        after_conv1 = seq_len - (kernel_size - 1)
        after_conv2 = after_conv1 - (kernel_size - 1)
        after_conv3 = after_conv2 - (kernel_size - 1)
        return after_conv3 * (n_filters * 3)

    def forward(self, target: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        target : torch.LongTensor  shape (batch_size, max_seq_len)

        Returns
        -------
        torch.Tensor  shape (batch_size, output_dim)
        """
        # Embedding: (B, L) → (B, L, embed_dim)
        x = self.embedding(target)

        # Permute cho Conv1d: (B, L, E) → (B, E, L)
        x = x.permute(0, 2, 1)

        x = torch.relu(self.bn1(self.conv1(x)))  # (B, n_filters, L1)
        x = torch.relu(self.bn2(self.conv2(x)))  # (B, n_filters*2, L2)
        x = torch.relu(self.bn3(self.conv3(x)))  # (B, n_filters*3, L3)

        # Flatten: (B, n_filters*3, L3) → (B, flatten_dim)
        x = x.view(x.size(0), -1)

        # FC projection
        x = self.fc(x)
        return x
