#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import multi_task_linear_predict  # noqa: E402

def main() -> None:
    n_samples, n_features, n_targets = 500, 8, 3
    x = np.ascontiguousarray(np.sin(np.arange(n_samples * n_features) * 0.03))
    weights = np.ascontiguousarray(
        np.cos(np.arange(n_features * n_targets) * 0.05) * 0.1
    )
    t0 = time.perf_counter()
    for _ in range(200):
        multi_task_linear_predict(x, weights, n_samples, n_features, n_targets)
    py_s = time.perf_counter() - t0
    try:
        import predicting_three_pollutants_at_once_multi_task_learning_for_power_plants_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(x, weights, n_samples, n_features, n_targets, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        multi_task_linear_predict(x, weights, n_samples, n_features, n_targets),
        np.asarray(
            rs.multi_task_linear_predict_py(
                x, weights, n_samples, n_features, n_targets
            )
        ),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
