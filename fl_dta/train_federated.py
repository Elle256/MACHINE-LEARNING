# -*- coding: utf-8 -*-
"""
train_federated.py
-------------------
Entry point chạy Federated Learning (FedAvg).

Usage:
    # IID
    python train_federated.py --dataset davis --partition iid --num_clients 5 --rounds 100

    # Non-IID (kinase-based)
    python train_federated.py --dataset davis --partition non_iid --num_clients 5 --rounds 100

    # Non-IID Dirichlet
    python train_federated.py --dataset davis --partition dirichlet --alpha 0.3 --num_clients 5 --rounds 100
"""

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from data.dataset import DTADataset, load_dataset
from data.partition import get_partition
from federation.server import FLServer
from models.deep_dta import build_model
from utils.logger import Logger, make_run_name
from utils.seed import set_seed


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", default="davis", choices=["davis", "kiba"])
    parser.add_argument("--data_dir", default="./data/raw")

    # FL hyperparams
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)

    # Partition
    parser.add_argument(
        "--partition",
        default="non_iid",
        choices=["iid", "non_iid", "dirichlet"],
    )
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha (chỉ dùng khi partition=dirichlet)")

    # Training
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--config", default="./configs/default.yaml")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)

    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.mode = "fedavg"  # chỉ dùng FedAvg
    set_seed(args.seed)

    # Load YAML config
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"] = args.dataset

    # ── Data ────────────────────────────────────────────────────────────────
    drug_smiles, protein_seqs, labels, meta_df = load_dataset(
        args.dataset, args.data_dir
    )

    # Train/test split (test set dùng chung cho evaluate global model)
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

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # ── Partition train set cho clients ─────────────────────────────────────
    # meta_df cần được filter theo train_idx cho Non-IID
    train_meta = meta_df.iloc[train_idx].reset_index(drop=False)
    train_meta.index = list(range(len(train_idx)))  # reindex 0..N_train-1

    client_indices = get_partition(
        strategy=args.partition,
        num_samples=len(train_idx),
        num_clients=args.num_clients,
        meta_df=train_meta,
        alpha=args.alpha,
        seed=args.seed,
    )

    # DTADataset dùng trực tiếp (không Subset) vì LocalUpdate sẽ tự Subset
    # → cần map local indices về global train indices
    global_client_indices = {
        cid: [train_idx[i] for i in local_idxs]
        for cid, local_idxs in client_indices.items()
    }

    # ── Model ───────────────────────────────────────────────────────────────
    global_model = build_model(cfg)
    n_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Logger ──────────────────────────────────────────────────────────────
    run_name = make_run_name(args)
    logger = Logger(
        results_dir=args.results_dir,
        run_name=run_name,
        use_wandb=args.use_wandb,
        project="fl-dta",
        config=vars(args),
    )

    # ── FL Server ───────────────────────────────────────────────────────────
    server = FLServer(
        args=args,
        global_model=global_model,
        train_dataset=full_dataset,   # full dataset; client dùng global indices
        test_dataset=test_dataset,
        client_indices=global_client_indices,
        logger=logger,
    )

    history = server.run()

    # ── Save final model ────────────────────────────────────────────────────
    ckpt_path = Path(args.results_dir) / f"{run_name}_final.pt"
    torch.save(server.global_model.state_dict(), ckpt_path)
    print(f"✓ Final model saved: {ckpt_path}")

    logger.finish()


if __name__ == "__main__":
    main()
