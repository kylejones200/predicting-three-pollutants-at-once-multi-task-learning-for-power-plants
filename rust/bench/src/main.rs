use predicting_three_pollutants_at_once_multi_task_learning_for_power_plants_core::multi_task_linear_predict;

fn main() {
    let n_samples = 500usize;
    let n_features = 8usize;
    let n_targets = 3usize;
    let x: Vec<f64> = (0..n_samples * n_features)
        .map(|i| (i as f64 * 0.03).sin())
        .collect();
    let weights: Vec<f64> = (0..n_features * n_targets)
        .map(|i| (i as f64 * 0.05).cos() * 0.1)
        .collect();
    for _ in 0..2000 {
        let _ = multi_task_linear_predict(&x, &weights, n_samples, n_features, n_targets);
    }
}
