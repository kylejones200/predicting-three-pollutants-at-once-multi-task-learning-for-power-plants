use predicting_three_pollutants_at_once_multi_task_learning_for_power_plants_core::multi_task_linear_predict;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn multi_task_linear_predict_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(multi_task_linear_predict(
        x.as_slice()?,
        weights.as_slice()?,
        n_samples,
        n_features,
        n_targets,
    )
    .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (x, weights, n_samples, n_features, n_targets, iterations=2_000))]
fn bench_kernel_py(
    x: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    iterations: usize,
) -> PyResult<f64> {
    let xb = x.as_slice()?.to_vec();
    let wb = weights.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = multi_task_linear_predict(&xb, &wb, n_samples, n_features, n_targets);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn predicting_three_pollutants_at_once_multi_task_learning_for_power_plants_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multi_task_linear_predict_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
