"""
Python code extracted from 05_multi_task_learning_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)
plants = pd.read_parquet("egrid_all_plants_1996-2023.parquet")
plants_2023 = plants[plants["data_year"] == 2023].copy()
co2 = pd.to_numeric(plants_2023["Plant annual CO2 emissions (tons)"], errors="coerce")
nox = pd.to_numeric(plants_2023["Plant annual NOx emissions (tons)"], errors="coerce")
so2 = pd.to_numeric(plants_2023["Plant annual SO2 emissions (tons)"], errors="coerce")
emissions_df = pd.DataFrame(
    {"log_co2": np.log1p(co2), "log_nox": np.log1p(nox), "log_so2": np.log1p(so2)}
).dropna()
corr = emissions_df.corr()
logger.info("Emissions Correlations:")
logger.info(corr)
sns.heatmap(
    corr,
    annot=True,
    cmap="RdYlGn",
    center=0,
    square=True,
    linewidths=2,
    cbar_kws={"shrink": 0.8},
)
plt.title("Pollutant Correlations: Why MTL Works")
plt.tight_layout()
plt.savefig("pollutant_correlations.png", dpi=150)


class _MLPForecaster(nn.Module):
    """MLP forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.ReLU(),
            nn.Linear(1, 1), nn.ReLU(),
            nn.Linear(1, 1), nn.ReLU(),
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.ReLU(),
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.ReLU(),
            nn.Linear(1, 1), nn.ReLU(),
            nn.Linear(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 64,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 10) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def build_mtl_model(input_dim, architecture="hard_sharing"):
    """
    Build multi-task learning model

    Parameters:
    - input_dim: Number of input features
    - architecture: 'hard_sharing' or 'soft_sharing'

    Returns:
    - Keras model with three outputs (CO2, NOx, SO2)
    """
    inputs = keras.Input(shape=(input_dim,), name="input_features")
    shared = layers.Dense(128, activation="relu", name="shared_1")(inputs)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(64, activation="relu", name="shared_2")(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(32, activation="relu", name="shared_3")(shared)
    shared = layers.BatchNormalization()(shared)
    co2_head = layers.Dense(16, activation="relu", name="co2_head")(shared)
    co2_output = layers.Dense(1, name="co2_output")(co2_head)
    nox_head = layers.Dense(16, activation="relu", name="nox_head")(shared)
    nox_output = layers.Dense(1, name="nox_output")(nox_head)
    so2_head = layers.Dense(16, activation="relu", name="so2_head")(shared)
    so2_output = layers.Dense(1, name="so2_output")(so2_head)
    model = keras.Model(
        inputs=inputs,
        outputs=[co2_output, nox_output, so2_output],
        name="mtl_emissions_predictor",
    )
    return model


model = build_mtl_model(input_dim=10)
model.summary()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_cols = [
    "Plant nameplate capacity (MW)",
    "Plant annual net generation (MWh)",
    "Plant annual heat input (MMBtu)",
    "Plant state abbreviation",
    "Plant primary fuel category",
]
X = plants_2023[feature_cols].copy()
X["capacity_mw"] = pd.to_numeric(X["Plant nameplate capacity (MW)"], errors="coerce")
X["generation_mwh"] = pd.to_numeric(
    X["Plant annual net generation (MWh)"], errors="coerce"
)
X["heat_input_mmbtu"] = pd.to_numeric(
    X["Plant annual heat input (MMBtu)"], errors="coerce"
)
X_encoded = pd.get_dummies(
    X[["Plant state abbreviation", "Plant primary fuel category"]], drop_first=True
)
X_features = pd.concat(
    [X[["capacity_mw", "generation_mwh", "heat_input_mmbtu"]], X_encoded], axis=1
)
y_co2 = emissions_df["log_co2"]
y_nox = emissions_df["log_nox"]
y_so2 = emissions_df["log_so2"]
common_idx = X_features.index.intersection(y_co2.index)
X_features = X_features.loc[common_idx]
y_co2 = y_co2.loc[common_idx]
y_nox = y_nox.loc[common_idx]
y_so2 = y_so2.loc[common_idx]
logger.info(f"Training on {len(X_features):,} plants")
X_train, X_test, y_co2_train, y_co2_test = train_test_split(
    X_features, y_co2, test_size=0.2, random_state=42
)
_, _, y_nox_train, y_nox_test = train_test_split(
    X_features, y_nox, test_size=0.2, random_state=42
)
_, _, y_so2_train, y_so2_test = train_test_split(
    X_features, y_so2, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
input_dim = X_train_scaled.shape[1]
mtl_model = build_mtl_model(input_dim)
,
    loss={"co2_output": "mse", "nox_output": "mse", "so2_output": "mse"},
    metrics={
        "co2_output": ["mae", "mse"],
        "nox_output": ["mae", "mse"],
        "so2_output": ["mae", "mse"],
    },
)
history = _train_torch(mtl_model, X_train_scaled, {"co2_output": y_co2_train),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ],
    verbose=1,
)
results = mtl_model.evaluate(
    X_test_scaled,
    {"co2_output": y_co2_test, "nox_output": y_nox_test, "so2_output": y_so2_test},
)
logger.info("\nMTL Model Performance:")
logger.info(f"CO2 MAE: {results[4]:.4f}")
logger.info(f"NOx MAE: {results[7]:.4f}")
logger.info(f"SO2 MAE: {results[10]:.4f}")


def build_single_task_model(input_dim):
    """Single-task model for comparison"""
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    ,
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


logger.info("\nTraining single-task baselines...")
co2_model = build_single_task_model(input_dim)
_train_torch(co2_model, X_train_scaled, y_co2_train)],
    verbose=0,
)
co2_baseline_mae = co2_model.evaluate(X_test_scaled, y_co2_test, verbose=0)[1]
nox_model = build_single_task_model(input_dim)
_train_torch(nox_model, X_train_scaled, y_nox_train)],
    verbose=0,
)
nox_baseline_mae = nox_model.evaluate(X_test_scaled, y_nox_test, verbose=0)[1]
so2_model = build_single_task_model(input_dim)
_train_torch(so2_model, X_train_scaled, y_so2_train)],
    verbose=0,
)
so2_baseline_mae = so2_model.evaluate(X_test_scaled, y_so2_test, verbose=0)[1]
logger.info("\nSingle-Task Baselines:")
logger.info(f"CO2 MAE: {co2_baseline_mae:.4f}")
logger.info(f"NOx MAE: {nox_baseline_mae:.4f}")
logger.info(f"SO2 MAE: {so2_baseline_mae:.4f}")
comparison = pd.DataFrame(
    {
        "Task": ["CO₂", "NOx", "SO₂"],
        "Single-Task MAE": [co2_baseline_mae, nox_baseline_mae, so2_baseline_mae],
        "MTL MAE": [results[4], results[7], results[10]],
    }
)
comparison["Improvement %"] = (
    (comparison["Single-Task MAE"] - comparison["MTL MAE"])
    / comparison["Single-Task MAE"]
    * 100
)
logger.info("\nMTL vs Single-Task Comparison:")
logger.info(comparison.to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35
ax.bar(
    x - width / 2,
    comparison["Single-Task MAE"],
    width,
    label="Single-Task",
    color="#e74c3c",
    alpha=0.8,
)
ax.bar(
    x + width / 2,
    comparison["MTL MAE"],
    width,
    label="Multi-Task",
    color="#2ecc71",
    alpha=0.8,
)
ax.set_xlabel("Pollutant", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Absolute Error (log scale)", fontsize=12, fontweight="bold")
ax.set_title("Multi-Task Learning Improves All Tasks", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Task"])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
for i, row in comparison.iterrows():
    ax.text(
        i,
        max(row["Single-Task MAE"], row["MTL MAE"]) + 0.02,
        f"+{row['Improvement %']:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="green",
    )
plt.tight_layout()
plt.savefig("mtl_vs_single_task.png", dpi=150)
mtl_model_weighted = build_mtl_model(input_dim)
,
    loss={"co2_output": "mse", "nox_output": "mse", "so2_output": "mse"},
    loss_weights={"co2_output": 2.0, "nox_output": 1.0, "so2_output": 1.0},
    metrics={"co2_output": ["mae"], "nox_output": ["mae"], "so2_output": ["mae"]},
)
history_weighted = _train_torch(mtl_model_weighted, X_train_scaled, {"co2_output": y_co2_train)],
    verbose=0,
)
results_weighted = mtl_model_weighted.evaluate(
    X_test_scaled,
    {"co2_output": y_co2_test, "nox_output": y_nox_test, "so2_output": y_so2_test},
    verbose=0,
)
logger.info("\nWeighted MTL (CO2 priority):")
logger.info(f"CO2 MAE: {results_weighted[2]:.4f} (improved!)")
logger.info(f"NOx MAE: {results_weighted[3]:.4f} (slightly worse)")
logger.info(f"SO2 MAE: {results_weighted[4]:.4f} (slightly worse)")
if mtl_mae > single_task_mae:
    logger.info("Tasks may be unrelated or need different architectures")


def build_soft_sharing_model(input_dim, n_tasks=3):
    """
    Soft parameter sharing: separate networks with L2 regularization
    encouraging similarity
    """
    inputs = keras.Input(shape=(input_dim,))
    task_networks = []
    for i in range(n_tasks):
        net = layers.Dense(
            128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
        )(inputs)
        net = layers.Dense(
            64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
        )(net)
        net = layers.Dense(32, activation="relu")(net)
        task_networks.append(net)
    co2_output = layers.Dense(1, name="co2_output")(task_networks[0])
    nox_output = layers.Dense(1, name="nox_output")(task_networks[1])
    so2_output = layers.Dense(1, name="so2_output")(task_networks[2])
    model = keras.Model(inputs=inputs, outputs=[co2_output, nox_output, so2_output])
    return model


mtl_model.save("mtl_emissions_predictor.h5")
loaded_model = keras.models.load_model("mtl_emissions_predictor.h5")
new_plant = X_test_scaled[:1]
co2_pred, nox_pred, so2_pred = _predict_torch(loaded_model, new_plant)
logger.info(f"Predicted CO2: {np.expm1(co2_pred[0][0]):,.0f} tons")
logger.info(f"Predicted NOx: {np.expm1(nox_pred[0][0]):,.0f} tons")
logger.info(f"Predicted SO2: {np.expm1(so2_pred[0][0]):,.0f} tons")
inputs = keras.Input(shape=(input_dim,), name="input_features")
shared = layers.Dense(128, activation="relu", name="shared_1")(inputs)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.3)(shared)
shared = layers.Dense(64, activation="relu", name="shared_2")(shared)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.3)(shared)
shared = layers.Dense(32, activation="relu", name="shared_3")(shared)
shared = layers.BatchNormalization()(shared)
co2_head = layers.Dense(16, activation="relu", name="co2_head")(shared)
co2_output = layers.Dense(1, name="co2_output")(co2_head)
nox_head = layers.Dense(16, activation="relu", name="nox_head")(shared)
nox_output = layers.Dense(1, name="nox_output")(nox_head)
so2_head = layers.Dense(16, activation="relu", name="so2_head")(shared)
so2_output = layers.Dense(1, name="so2_output")(so2_head)
model = keras.Model(
    inputs=inputs,
    outputs=[co2_output, nox_output, so2_output],
    name="mtl_emissions_predictor",
)
return model
Model: "mtl_emissions_predictor"
__________________________________________________________________________________________________
__________________________________________________________________________________________________
optimizer = (keras.optimizers.Adam(learning_rate=0.001),)
loss = ({"co2_output": "mse", "nox_output": "mse", "so2_output": "mse"},)
metrics = {
    "co2_output": ["mae", "mse"],
    "nox_output": ["mae", "mse"],
    "so2_output": ["mae", "mse"],
}
(X_train_scaled,)
({"co2_output": y_co2_train, "nox_output": y_nox_train, "so2_output": y_so2_train},)
validation_split = (0.2,)
epochs = (50,)
batch_size = (64,)
callbacks = (
    [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ],
)
verbose = 1
"Single-task model for comparison"
model = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)
,
    loss="mse",
    metrics=["mae", "mse"],
)
return model
ax.text(
    i,
    max(row["Single-Task MAE"], row["MTL MAE"]) + 0.02,
    f"+{row['Improvement %']:.1f}%",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color="green",
)
optimizer = (keras.optimizers.Adam(learning_rate=0.001),)
loss = ({"co2_output": "mse", "nox_output": "mse", "so2_output": "mse"},)
loss_weights = ({"co2_output": 2.0, "nox_output": 1.0, "so2_output": 1.0},)
metrics = {"co2_output": ["mae"], "nox_output": ["mae"], "so2_output": ["mae"]}
(X_train_scaled,)
({"co2_output": y_co2_train, "nox_output": y_nox_train, "so2_output": y_so2_train},)
validation_split = (0.2,)
epochs = (50,)
batch_size = (64,)
callbacks = ([keras.callbacks.EarlyStopping(patience=10)],)
verbose = 0
(X_test_scaled,)
({"co2_output": y_co2_test, "nox_output": y_nox_test, "so2_output": y_so2_test},)
verbose = 0
logger.info("Tasks may be unrelated or need different architectures")
"\nSoft parameter sharing: separate networks with L2 regularization\nencouraging similarity\n"
inputs = keras.Input(shape=(input_dim,))
task_networks = []
for i in range(n_tasks):
    net = layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
    )(inputs)
    net = layers.Dense(
        64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)
    )(net)
    net = layers.Dense(32, activation="relu")(net)
    task_networks.append(net)
co2_output = layers.Dense(1, name="co2_output")(task_networks[0])
nox_output = layers.Dense(1, name="nox_output")(task_networks[1])
so2_output = layers.Dense(1, name="so2_output")(task_networks[2])
model = keras.Model(inputs=inputs, outputs=[co2_output, nox_output, so2_output])
return model
