"""Multi-task linear prediction: y_hat = X @ W (row-major)."""

from __future__ import annotations

import numpy as np


def multi_task_linear_predict(
    x: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    n_features: int,
    n_targets: int,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    out = np.zeros(n_samples * n_targets, dtype=float)
    for i in range(n_samples):
        for t in range(n_targets):
            total = 0.0
            for f in range(n_features):
                total += x_arr[i * n_features + f] * w_arr[f * n_targets + t]
            out[i * n_targets + t] = total
    return out
