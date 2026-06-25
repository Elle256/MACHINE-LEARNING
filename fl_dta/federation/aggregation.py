from typing import List, Dict
import numpy as np


def fedavg_aggregate(
    local_weights: List[Dict[str, np.ndarray]],
    client_sizes: List[int],
) -> Dict[str, np.ndarray]:
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
