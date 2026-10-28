import csv
import uuid
import os
from dataclasses import dataclass
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
    "generic",
    "n_qubits",
    "layers",
    "anzats_reps",
    "n_params",
    "n_train_samples",
    "max_steps",
    "batch_size",
    "n_expvals",
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

@dataclass
class TrainingResult:
    """Everything record_training needs to persist one run, with no file
    I/O of its own"""
    circuit_num: int
    generic: bool
    n_qubits: int
    layers: int
    anzats_reps: int
    n_params: int
    n_train_samples: int
    max_steps: int
    batch_size: int
    n_expvals: int
    final_weights: "torch.Tensor"
    cst: "torch.Tensor"
    notes: str = ""


def run_training(
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
    train_fn = train,
    generic: bool = False,
) -> TrainingResult:
    """Build the model and run `train` -- pure compute, no file I/O. Split
    out of train_and_record so the expensive part (this) can run in a
    parallel worker process while a single, non-concurrent caller does all
    the disk writing afterward (record_training) -- concurrent appends to
    the same experiments.csv from multiple OS processes aren't safe, so
    only ever call record_training from one process at a time against a
    given `path`. See run.py's `--n-workers` for the parallel caller.
    """
    model, weights = build_model(circuit_num, n_qubits, layers, anzats_reps,
                                  measuring_qubit=n_qubits - 1, generic=generic)

    final_weights, cst = train_fn(
        model, weights, x, target_y,
        max_steps=max_steps,
        batch_size=batch_size,
        display_step=display_step,
        display=False,
    )

    return TrainingResult(
        circuit_num=circuit_num,
        generic=generic,
        n_qubits=n_qubits,
        layers=layers,
        anzats_reps=anzats_reps,
        # weight_tensor_shape allocates a rectangular tensor, but several
        # circuits don't read all of it — count what's
        # actually trainable rather than the raw tensor size.
        n_params=layers * n_trainable(circuit_num, n_qubits, anzats_reps, generic=generic),
        n_train_samples=len(x),
        max_steps=max_steps,
        batch_size=batch_size,
        # How many times the (simulated) quantum circuit's expectation value
        # was actually estimated over the whole run.
        n_expvals=getattr(model, "n_expvals", None),
        final_weights=final_weights,
        cst=cst,
        notes=notes,
    )


def record_training(
    result: TrainingResult,
    experiments_csv: str = EXPERIMENTS_CSV,
    path: str = "results/",
) -> str:
    """Persist a TrainingResult (from run_training, possibly computed in a
    different process) to experiments.csv and costs/{id}.npy. See
    run_training's docstring for why this is a separate call -- only call
    this from one process at a time against a given `path`.

    Returns
    -------
    experiment_id : str
        UUID of the recorded experiment.
    """
    experiments_csv = experiments_csv if path is None else os.path.join(path, experiments_csv)

    _ensure_csv(experiments_csv, EXPERIMENTS_FIELDS)

    experiment_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat(timespec="seconds")
    cst = result.cst

    exp_row = {
        "id":               experiment_id,
        "timestamp":        timestamp,
        "circuit_num":      result.circuit_num,
        "generic":          result.generic,
        "n_qubits":         result.n_qubits,
        "layers":           result.layers,
        "anzats_reps":      result.anzats_reps,
        "n_params":         result.n_params,
        "n_train_samples":  result.n_train_samples,
        "max_steps":        result.max_steps,
        "batch_size":       result.batch_size,
        "n_expvals":        result.n_expvals,
        # cst[0]/cst[-1]: the first/last step's own (pre-update) batch loss --
        # see train()'s docstring for why it's the batch loss, not a separate
        # full-dataset evaluation.
        "initial_cost":     round(float(cst[0].item()), 8),
        "final_cost":       round(float(cst[-1].item()), 8),
        "notes":            result.notes,
    }
    _append_row(experiments_csv, EXPERIMENTS_FIELDS, exp_row)

    # cst holds every one of the max_steps optimizer steps' own batch loss --
    # no separate pre-training slot to skip any more (see train()'s docstring).
    save_cost_curve(experiment_id, cst.detach().numpy(), path=path)

    print(f"[tracker] Experiment {experiment_id} saved "
          f"(circuit={result.circuit_num}, qubits={result.n_qubits}, layers={result.layers}, "
          f"final_cost={exp_row['final_cost']:.6f})")

    return experiment_id


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
    generic: bool = False,
):
    """
    Run `run_training` then `record_training` -- the single-call,
    single-process convenience most callers want. See their own docstrings
    for the parallel-friendly split (run.py's `--n-workers` uses it
    directly instead of this wrapper).

    Returns
    -------
    experiment_id : str
        UUID of the recorded experiment.
    final_weights : torch.Tensor
    cost_history : torch.Tensor  shape (max_steps,)
    """
    result = run_training(
        x, target_y, circuit_num, n_qubits, layers, anzats_reps,
        max_steps=max_steps, batch_size=batch_size, display_step=display_step,
        notes=notes, train_fn=train_fn, generic=generic,
    )
    experiment_id = record_training(result, experiments_csv=experiments_csv, path=path)
    return experiment_id, result.final_weights, result.cst


