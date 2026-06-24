# -*- coding: utf-8 -*-
"""
models/deep_dta.py
------------------
DeepDTA model tổng hợp: Drug GNN encoder + Protein CNN encoder + FC layers.

Input  : (drug_batch, protein_tensor)
Output : affinity score (scalar per sample)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .drug_encoder import DrugGNNEncoder
from .protein_encoder import ProteinCNNEncoder


class DeepDTA(nn.Module):
    """
    Drug-Target Affinity prediction model.

    Architecture:
        Drug (graph)   → GNN → drug_embed   (input_dim,)
        Protein (seq)  → CNN → protein_embed (input_dim,)
        concat         → FC layers           → affinity scalar

    Parameters
    ----------
    drug_encoder : DrugGNNEncoder
    protein_encoder : ProteinCNNEncoder
    input_dim : int
        Embedding dim (phải khớp output_dim của cả 2 encoders).
    dropout : float
    n_output : int
        1 cho regression (MSE), >1 cho classification.
    """

    def __init__(
        self,
        drug_encoder: DrugGNNEncoder,
        protein_encoder: ProteinCNNEncoder,
        input_dim: int = 128,
        dropout: float = 0.2,
        n_output: int = 1,
    ):
        super().__init__()

        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.n_output = n_output

        # Fusion MLP
        self.fc = nn.Sequential(
            nn.Linear(2 * input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_output),
        )

    def forward(
        self,
        drug_batch: Batch,
        protein: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        drug_batch : torch_geometric.data.Batch
        protein    : torch.LongTensor  (B, max_seq_len)

        Returns
        -------
        torch.Tensor  (B, n_output)  hoặc (B,) nếu n_output=1
        """
        drug_feat = self.drug_encoder(drug_batch)        # (B, input_dim)
        protein_feat = self.protein_encoder(protein)      # (B, input_dim)

        combined = torch.cat([drug_feat, protein_feat], dim=1)  # (B, 2*input_dim)
        out = self.fc(combined)                                   # (B, n_output)

        if self.n_output == 1:
            out = out.squeeze(-1)  # (B,)

        return out


def build_model(cfg: dict) -> DeepDTA:
    """
    Khởi tạo model từ config dict.

    Parameters
    ----------
    cfg : dict  (từ configs/default.yaml)

    Returns
    -------
    DeepDTA instance
    """
    drug_cfg = cfg["drug_encoder"]
    protein_cfg = cfg["protein_encoder"]
    model_cfg = cfg["model"]

    drug_encoder = DrugGNNEncoder(
        num_features=drug_cfg["num_features_drug"],
        hidden_dim=drug_cfg["hidden_dim"],
        output_dim=drug_cfg["embed_dim"],
        num_layers=drug_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )

    protein_encoder = ProteinCNNEncoder(
        num_features_protein=protein_cfg["num_features_protein"],
        embed_dim=protein_cfg["embed_dim"],
        n_filters=protein_cfg["n_filters"],
        max_seq_len=protein_cfg["max_seq_len"],
        output_dim=protein_cfg["output_dim"],
        dropout=model_cfg["dropout"],
    )

    model = DeepDTA(
        drug_encoder=drug_encoder,
        protein_encoder=protein_encoder,
        input_dim=model_cfg["input_dim"],
        dropout=model_cfg["dropout"],
        n_output=model_cfg["n_output"],
    )

    return model
