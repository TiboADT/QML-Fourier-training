import csv
import uuid
import os
from datetime import datetime

import numpy as np

from functions import train, build_model
from circuits import n_trainable

# ------------------------------------------------------------------
# Paths (override before importing if needed)
# ------------------------------------------------------------------
EXPERIMENTS_CSV = "experiments.csv"
COSTS_DIR       = "costs"

EXPERIMENTS_FIELDS = [
    "id",
    "timestamp",
    "circuit_num",
    "n_qubits",
    "layers",
    "anzats_reps",
    "n_params",
    "n_train_samples",
    "max_steps",
    "batch_size",
    "final_cost",
    "initial_cost",
    "notes",
]


def _ensure_csv(path: str, fieldnames: list[str]) -> None:
    """Create the CSV with a header row if it does not exist yet."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def _append_row(path: str, fieldnames: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def load_experiments(experiments_csv: str = EXPERIMENTS_CSV, path: str = None):
    """Return all experiment summary rows as a list of dicts."""
    if path is not None:
        experiments_csv = os.path.join(path, experiments_csv)
    _ensure_csv(experiments_csv, EXPERIMENTS_FIELDS)
    with open(experiments_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def save_cost_curve(experiment_id: str, costs, path: str = "results/") -> None:
    """Save one experiment's cost curve as costs/{experiment_id}.npy.
    """
    directory = os.path.join(path, COSTS_DIR) if path else COSTS_DIR
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, f"{experiment_id}.npy"), np.asarray(costs, dtype=np.float64))


def load_cost_curve(experiment_id: str, path: str = "results/") -> np.ndarray:
    """Load a single experiment's cost curve."""
    directory = os.path.join(path, COSTS_DIR) if path else COSTS_DIR
    return np.load(os.path.join(directory, f"{experiment_id}.npy"))


def load_costs(path: str = "results/"):
    """All recorded cost curves as a list of dicts: experiment_id, n_steps, costs."""
    directory = os.path.join(path, COSTS_DIR) if path else COSTS_DIR
    if not os.path.isdir(directory):
        return []
    parsed = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".npy"):
            continue
        costs = np.load(os.path.join(directory, filename))
        parsed.append({
            "experiment_id": filename[:-len(".npy")],
            "n_steps": len(costs),
            "costs": costs.tolist(),
        })
    return parsed



# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train_and_record(
    x,
    target_y,
    circuit_num: int,
    n_qubits: int,
    layers: int,
    anzats_reps: int,
    max_steps: int = 200,
    batch_size: int = 100,
    display_step: int = 10,
    notes: str = "",
    experiments_csv: str = EXPERIMENTS_CSV,
    train_fn = train,
    path: str = "results/",
):
    """
    Run `train`, then persist metadata to experiments.csv and the cost curve
    to costs/{experiment_id}.npy.

    Parameters
    ----------
    model, weights, x, target_y :
        Passed straight through to `train`.
    circuit_num : int
        The circuit index (1-19) used in this experiment.
    n_qubits : int
        Number of qubits.
    reps : int
        Number of repetitions of the ansatz.
    max_steps, batch_size, display_step :
        Training hyper-parameters forwarded to `train`.
    notes : str
        Any free-text annotation to attach to this run.
    experiments_csv : str
        Path to the experiments summary CSV.

    Returns
    -------
    experiment_id : str
        UUID of the recorded experiment.
    final_weights : torch.Tensor
    cost_history : torch.Tensor  shape (max_steps,)
    """
    experiments_csv = experiments_csv if path is None else os.path.join(path, experiments_csv)

    _ensure_csv(experiments_csv, EXPERIMENTS_FIELDS)

    experiment_id = str(uuid.uuid4())
    timestamp     = datetime.now().isoformat(timespec="seconds")

    # ---- run training ------------------------------------------------
    model,weights = build_model(circuit_num, n_qubits, layers, anzats_reps, measuring_qubit=n_qubits-1)

    final_weights, cst = train_fn(
        model, weights, x, target_y,
        max_steps=max_steps,
        batch_size=batch_size,
        display_step=display_step,
        display=False,
    )

    # ---- write experiment summary row --------------------------------
    exp_row = {
        "id":               experiment_id,
        "timestamp":        timestamp,
        "circuit_num":      circuit_num,
        "n_qubits":         n_qubits,
        "layers":            layers,
        "anzats_reps":             anzats_reps,
        # weight_tensor_shape allocates a rectangular tensor, but several
        # circuits don't read all of it — count what's
        # actually trainable rather than the raw tensor size.
        "n_params":         layers * n_trainable(circuit_num, n_qubits, anzats_reps),
        "n_train_samples":  len(x),
        "max_steps":        max_steps,
        "batch_size":       batch_size,
        "initial_cost":     round(float(cst[0].item()), 8),
        "final_cost":       round(float(cst[-1].item()), 8),
        "notes":            notes,
    }
    _append_row(experiments_csv, EXPERIMENTS_FIELDS, exp_row)

    # ---- save the cost curve -------------------------------------------
    # cst[0] is the pre-training cost (already recorded as initial_cost above);
    # cst[1:] holds the cost after each of the max_steps optimizer steps.
    save_cost_curve(experiment_id, cst[1:].detach().numpy(), path=path)

    print(f"[tracker] Experiment {experiment_id} saved "
          f"(circuit={circuit_num}, qubits={n_qubits}, layers={layers}, "
          f"final_cost={exp_row['final_cost']:.6f})")

    return experiment_id, final_weights, cst


