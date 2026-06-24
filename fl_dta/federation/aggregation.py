# -*- coding: utf-8 -*-
"""
federation/aggregation.py
--------------------------
FedAvg weighted aggregation của model weights.
"""

from typing import List, Dict
import numpy as np


def fedavg_aggregate(
    local_weights: List[Dict[str, np.ndarray]],
    client_sizes: List[int],
) -> Dict[str, np.ndarray]:
    """
    Weighted average của state_dicts theo số samples mỗi client.

    Parameters
    ----------
    local_weights : List of state_dict (key → numpy array)
    client_sizes  : Số samples của mỗi client

    Returns
    -------
    Aggregated state_dict
    """
    total = sum(client_sizes)
    keys = local_weights[0].keys()

    agg = {}
    for key in keys:
        weighted = sum(
            local_weights[i][key] * (client_sizes[i] / total)
            for i in range(len(local_weights))
        )
        agg[key] = weighted

    return agg


def apply_dropout(selected_clients: List[int], dropout_rate: float) -> List[int]:
    """Simulate client dropout. Giữ ít nhất 1 client."""
    if dropout_rate <= 0.0:
        return selected_clients

    survived = [c for c in selected_clients if np.random.random() > dropout_rate]
    if len(survived) == 0:
        survived = [selected_clients[np.random.randint(len(selected_clients))]]
    return survived
