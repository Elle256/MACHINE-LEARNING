from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# IID Partition

def partition_iid(num_samples: int, num_clients: int, seed: int = 42,) -> Dict[int, List[int]]:

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples).tolist()

    client_indices: Dict[int, List[int]] = {}
    chunk = num_samples // num_clients
    for cid in range(num_clients):
        start = cid * chunk
        end = start + chunk if cid < num_clients - 1 else num_samples
        client_indices[cid] = indices[start:end]

    return client_indices

# Non-IID Partition

def partition_non_iid_dirichlet(
    meta_df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:

    rng = np.random.default_rng(seed)

    families = meta_df["kinase_family"].unique()
    family_to_int = {f: i for i, f in enumerate(families)}
    labels = meta_df["kinase_family"].map(family_to_int).values
    n_classes = len(families)

    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for cls in range(n_classes):
        cls_mask = np.where(labels == cls)[0]
        rng.shuffle(cls_mask)

        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        proportions = (proportions / proportions.sum()).cumsum()

        splits = (proportions * len(cls_mask)).astype(int)
        splits[-1] = len(cls_mask)  # đảm bảo không mất sample

        prev = 0
        for cid, end in enumerate(splits):
            client_indices[cid].extend(cls_mask[prev:end].tolist())
            prev = end
    for cid in client_indices:
        rng.shuffle(client_indices[cid])
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
    rng = np.random.default_rng(seed)
    all_indices = list(meta_df.index)

    for cid in client_indices:
        if len(client_indices[cid]) < min_samples:
            extra = rng.choice(all_indices, size=min_samples, replace=False).tolist()
            client_indices[cid] = list(set(client_indices[cid]) | set(extra))

    return client_indices

def get_partition(
    strategy: str,
    num_samples: int,
    num_clients: int,
    meta_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
 
    if strategy == "iid":
        return partition_iid(num_samples, num_clients, seed)
    elif strategy == "dirichlet":
        if meta_df is None:
            raise ValueError("dirichlet")
        return partition_non_iid_dirichlet(meta_df, num_clients, alpha, seed)
    else:
        raise ValueError(f"Partition strategy error")
