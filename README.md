# Repository

Companion code for a Medium article.

## Business context

*Using shared neural network architectures to simultaneously predict CO₂, NOx, and SO₂ emissions with better accuracy than single-task models*

You need to predict CO₂ emissions from power plants. You build a model. It works.

Now you're maintaining three separate models, training them separately, deploying them separately. And here's the kicker: all three pollutants are closely related. Coal plants emit all three. Gas plants emit mostly CO₂ and NOx. The combustion chemistry is linked.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — multi-task linear prediction. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p predicting_three_pollutants_at_once_multi_task_learning_for_power_plants_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).