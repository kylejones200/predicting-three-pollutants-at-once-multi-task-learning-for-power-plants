//! Multi-task linear prediction: y_hat = X @ W (row-major).

pub fn multi_task_linear_predict(
    x: &[f64],
    weights: &[f64],
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
) -> Vec<f64> {
    assert_eq!(x.len(), n_samples * n_features);
    assert_eq!(weights.len(), n_features * n_targets);
    let mut out = vec![0.0; n_samples * n_targets];
    for i in 0..n_samples {
        for t in 0..n_targets {
            let mut sum = 0.0;
            for f in 0..n_features {
                sum += x[i * n_features + f] * weights[f * n_targets + t];
            }
            out[i * n_targets + t] = sum;
        }
    }
    out
}
