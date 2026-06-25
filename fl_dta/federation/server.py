from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import DTADataset
from federation.aggregation import fedavg_aggregate, apply_dropout
from federation.client import LocalUpdate
from utils.metrics import compute_metrics


class FLServer:
    def __init__(
        self,
        args,
        global_model: nn.Module,
        train_dataset: DTADataset,
        test_dataset: DTADataset,
        client_indices: Dict[int, List[int]],
        logger=None,
    ):
        self.args = args
        self.device = torch.device(args.device)
        self.global_model = global_model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_indices = client_indices
        self.num_clients = len(client_indices)
        self.logger = logger

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=DTADataset.collate_fn,
        )
        self.history: Dict[str, List] = {
            "round": [],
            "train_loss": [],
            "test_mse": [],
            "test_ci": [],
            "test_rm2": [],
        }

    # Main training loop 

    def run(self) -> Dict[str, List]:
        args = self.args
        num_rounds = args.rounds
        fraction = getattr(args, "fraction_fit", 1.0)
        dropout_rate = getattr(args, "dropout_rate", 0.0)

        print(f"\n{'='*60}")
        print(f"  FedAvg | {self.num_clients} clients | {num_rounds} rounds")
        print(f"  Dataset: {args.dataset} | Partition: {args.partition}")
        print(f"{'='*60}\n")

        for rnd in range(1, num_rounds + 1):
            selected = self._select_clients(fraction)
            selected = apply_dropout(selected, dropout_rate)

            local_weights_list = []
            local_losses = []
            client_sizes = []

            for cid in tqdm(selected, desc=f"Round {rnd}/{num_rounds}",leave=False):
                client = LocalUpdate(
                    args=args,
                    dataset=self.train_dataset,
                    idxs=self.client_indices[cid],
                    client_id=cid,
                )
                w, loss = client.train(self.global_model)
                local_weights_list.append(w)
                local_losses.append(loss)
                client_sizes.append(len(client))

            # Aggregate
            agg_weights = fedavg_aggregate(local_weights_list, client_sizes)
            self._load_numpy_weights(agg_weights)

            # Evaluate global model
            avg_train_loss = float(np.mean(local_losses))
            metrics = self._evaluate()

            # Log
            self.history["round"].append(rnd)
            self.history["train_loss"].append(avg_train_loss)
            self.history["test_mse"].append(metrics["mse"])
            self.history["test_ci"].append(metrics["ci"])
            self.history["test_rm2"].append(metrics["rm2"])

            if self.logger:
                self.logger.log_round(rnd, avg_train_loss, metrics)

            if rnd % getattr(args, "log_every", 10) == 0 or rnd == 1:
                print(
                    f"Round {rnd:>3}/{num_rounds} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"MSE: {metrics['mse']:.4f} | "
                    f"CI: {metrics['ci']:.4f} | "
                    f"rm²: {metrics['rm2']:.4f} | "
                    f"Clients: {selected}"
                )
        return self.history

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _select_clients(self, fraction: float) -> List[int]:
        k = max(1, int(self.num_clients * fraction))
        return random.sample(list(self.client_indices.keys()), k)

    def _load_numpy_weights(self, numpy_state_dict: Dict[str, np.ndarray]):
        torch_state = {
            k: torch.tensor(v) for k, v in numpy_state_dict.items()
        }
        self.global_model.load_state_dict(torch_state)

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.global_model.eval()
        all_preds, all_labels = [], []

        for drug_batch, protein, labels in self.test_loader:
            drug_batch = drug_batch.to(self.device)
            protein = protein.to(self.device)
            preds = self.global_model(drug_batch, protein)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        self.global_model.train()
        return compute_metrics(all_labels, all_preds)
