# -*- coding: utf-8 -*-
"""
data/partition.py
-----------------
Chiến lược phân chia dữ liệu cho Federated Learning:

  IID      : Shuffle rồi chia đều cho các client.
  Non-IID  : Mỗi client chuyên một hoặc một số kinase family
             (mô phỏng thực tế: mỗi lab dược chuyên 1 lĩnh vực).
             Hỗ trợ thêm Dirichlet-based non-IID.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# IID Partition
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Phân chia IID: shuffle indices rồi chia đều.

    Returns
    -------
    client_indices : dict {client_id: [sample_indices]}
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples).tolist()

    client_indices: Dict[int, List[int]] = {}
    chunk = num_samples // num_clients
    for cid in range(num_clients):
        start = cid * chunk
        end = start + chunk if cid < num_clients - 1 else num_samples
        client_indices[cid] = indices[start:end]

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# Non-IID Partition: Kinase Family–based
# ─────────────────────────────────────────────────────────────────────────────

def partition_non_iid_kinase(
    meta_df: pd.DataFrame,
    num_clients: int,
    seed: int = 42,
    min_samples_per_client: int = 50,
) -> Dict[int, List[int]]:
    """
    Non-IID partition dựa trên kinase family.

    Mỗi client được assign chủ yếu 1–2 kinase families,
    mô phỏng lab dược chuyên về một nhóm kinase cụ thể.

    Parameters
    ----------
    meta_df : DataFrame với cột 'kinase_family' và index = sample index
    num_clients : Số clients
    seed : Random seed

    Returns
    -------
    client_indices : dict {client_id: [sample_indices]}
    """
    rng = np.random.default_rng(seed)

    families = meta_df["kinase_family"].unique().tolist()
    rng.shuffle(families)

    # Gom samples theo family
    family_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in meta_df.iterrows():
        family_to_indices[row["kinase_family"]].append(idx)

    # Phân bổ families cho clients (round-robin)
    client_families: Dict[int, List[str]] = defaultdict(list)
    for i, family in enumerate(families):
        client_families[i % num_clients].append(family)

    # Tập hợp indices cho mỗi client
    client_indices: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        idxs = []
        for fam in client_families[cid]:
            idxs.extend(family_to_indices[fam])
        rng.shuffle(idxs)
        client_indices[cid] = idxs

    # Đảm bảo mỗi client có đủ samples
    client_indices = _ensure_min_samples(
        client_indices, meta_df, min_samples_per_client, seed
    )

    _print_partition_stats(client_indices, meta_df)
    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# Non-IID Partition: Dirichlet-based (tổng quát hơn)
# ─────────────────────────────────────────────────────────────────────────────

def partition_non_iid_dirichlet(
    meta_df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Non-IID partition bằng Dirichlet distribution.

    alpha nhỏ → heterogeneity cao hơn (non-IID mạnh hơn).
    alpha → ∞ → tiệm cận IID.

    Dùng protein_id làm "class label" để phân phối.
    """
    rng = np.random.default_rng(seed)

    # Dùng kinase_family làm class
    families = meta_df["kinase_family"].unique()
    family_to_int = {f: i for i, f in enumerate(families)}
    labels = meta_df["kinase_family"].map(family_to_int).values
    n_classes = len(families)

    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for cls in range(n_classes):
        cls_mask = np.where(labels == cls)[0]
        rng.shuffle(cls_mask)

        # Dirichlet proportions
        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        proportions = (proportions / proportions.sum()).cumsum()

        splits = (proportions * len(cls_mask)).astype(int)
        splits[-1] = len(cls_mask)  # đảm bảo không mất sample

        prev = 0
        for cid, end in enumerate(splits):
            client_indices[cid].extend(cls_mask[prev:end].tolist())
            prev = end

    # Shuffle từng client
    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    _print_partition_stats(client_indices, meta_df)
    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_min_samples(
    client_indices: Dict[int, List[int]],
    meta_df: pd.DataFrame,
    min_samples: int,
    seed: int,
) -> Dict[int, List[int]]:
    """Nếu client có quá ít samples, sample thêm từ pool chung."""
    rng = np.random.default_rng(seed)
    all_indices = list(meta_df.index)

    for cid in client_indices:
        if len(client_indices[cid]) < min_samples:
            extra = rng.choice(all_indices, size=min_samples, replace=False).tolist()
            client_indices[cid] = list(set(client_indices[cid]) | set(extra))

    return client_indices


def _print_partition_stats(
    client_indices: Dict[int, List[int]],
    meta_df: pd.DataFrame,
):
    """In thống kê phân chia để debug."""
    print("\n─── Partition Statistics ───────────────────────────────")
    for cid, idxs in client_indices.items():
        families = meta_df.loc[idxs, "kinase_family"].value_counts().to_dict()
        print(f"  Client {cid}: {len(idxs):>5} samples | families: {families}")
    print("────────────────────────────────────────────────────────\n")


def get_partition(
    strategy: str,
    num_samples: int,
    num_clients: int,
    meta_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Factory function chọn partition strategy.

    strategy : "iid" | "non_iid" | "dirichlet"
    """
    if strategy == "iid":
        return partition_iid(num_samples, num_clients, seed)
    elif strategy == "non_iid":
        if meta_df is None:
            raise ValueError("non_iid partition cần meta_df")
        return partition_non_iid_kinase(meta_df, num_clients, seed)
    elif strategy == "dirichlet":
        if meta_df is None:
            raise ValueError("dirichlet partition cần meta_df")
        return partition_non_iid_dirichlet(meta_df, num_clients, alpha, seed)
    else:
        raise ValueError(f"Partition strategy không hỗ trợ: {strategy}")
