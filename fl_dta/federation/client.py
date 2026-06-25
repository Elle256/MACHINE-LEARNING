# -*- coding: utf-8 -*-
"""
federation/client.py
---------------------
FL Client: local training trên dữ liệu riêng của mỗi client.
Chỉ giữ FedAvg (bỏ FedProx).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import DTADataset


class LocalUpdate:
    def __init__(
        self,
        args,
        dataset: DTADataset,
        idxs: List[int],
        client_id: int = 0,
    ):
        self.args = args
        self.client_id = client_id
        self.device = torch.device(args.device)

        subset = Subset(dataset, idxs)
        self.loader = DataLoader(
            subset,
            batch_size=args.local_bs,
            shuffle=True,
            collate_fn=DTADataset.collate_fn,
            drop_last=len(subset) > args.local_bs,  
        )

        self.loss_fn = nn.MSELoss()


    def train(
        self, global_model: nn.Module
    ) -> Tuple[Dict[str, np.ndarray], float]:

        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
        )

        epoch_losses = []

        for _ in range(self.args.local_epochs):
            batch_losses = []

            for drug_batch, protein, labels in self.loader:
                drug_batch = drug_batch.to(self.device)
                protein = protein.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                preds = model(drug_batch, protein)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            if batch_losses:
                epoch_losses.append(np.mean(batch_losses))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        state_numpy = {
            k: v.cpu().numpy()
            for k, v in model.state_dict().items()
        }

        return state_numpy, avg_loss

    def __len__(self) -> int:
        return len(self.loader.dataset)
