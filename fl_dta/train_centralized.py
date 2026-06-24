# -*- coding: utf-8 -*-
"""
train_centralized.py
---------------------
Baseline: train toàn bộ data tập trung (không FL).
Kết quả này là upper bound để so sánh với FL.

Usage:
    python train_centralized.py --dataset davis --epochs 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import DTADataset, load_dataset
from models.deep_dta import build_model
from utils.logger import Logger, make_run_name
from utils.metrics import compute_metrics
from utils.seed import set_seed


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="davis", choices=["davis", "kiba"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--config", default="./configs/default.yaml")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for drug_batch, protein, labels in loader:
        drug_batch = drug_batch.to(device)
        protein = protein.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(drug_batch, protein)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for drug_batch, protein, labels in loader:
        drug_batch = drug_batch.to(device)
        protein = protein.to(device)
        preds = model(drug_batch, protein)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    return compute_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Load config YAML
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override cfg với CLI args
    cfg["dataset"] = args.dataset

    print(f"\n[Centralized] Dataset={args.dataset} | Device={args.device}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    drug_smiles, protein_seqs, labels, meta_df = load_dataset(
        args.dataset, args.data_dir
    )

    indices = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed
    )

    cache_dir = Path(args.data_dir) / args.dataset / "cache"
    full_dataset = DTADataset(
        drug_smiles, protein_seqs, labels,
        max_protein_len=cfg["protein_encoder"]["max_seq_len"],
        cache_dir=str(cache_dir),
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DTADataset.collate_fn,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DTADataset.collate_fn,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    loss_fn = nn.MSELoss()

    # ── Logger ──────────────────────────────────────────────────────────────
    args.mode = "centralized"
    args.num_clients = 1
    args.partition = "full"
    run_name = make_run_name(args)

    logger = Logger(
        results_dir=args.results_dir,
        run_name=run_name,
        use_wandb=args.use_wandb,
        project="fl-dta",
        config=vars(args),
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Centralized Training | {args.epochs} epochs")
    print(f"{'='*55}\n")

    best_mse = float("inf")
    ckpt_path = Path(args.results_dir) / f"{run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, test_loader, device)
        scheduler.step(metrics["mse"])

        logger.log_epoch(epoch, train_loss, metrics)

        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            torch.save(model.state_dict(), ckpt_path)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"MSE: {metrics['mse']:.4f} | "
                f"CI: {metrics['ci']:.4f} | "
                f"rm²: {metrics['rm2']:.4f}"
            )

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n✓ Centralized training done. Best MSE: {best_mse:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    logger.finish()


if __name__ == "__main__":
    main()
