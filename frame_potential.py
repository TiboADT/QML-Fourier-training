"""
Frame potential estimation, computed directly from the PennyLane circuit
definitions in circuits.py.

Computing the frame potential from the exact same circuit_set(num)
function used for training makes that whole class of bug structurally
impossible.

Design : a frozen `Estimate` dataclass that behaves like an
accumulator (`estimate_a + estimate_b` pools two independent Monte Carlo
batches), and four small functions :
    sample_unitaries          — batched unitary construction (GPU-friendly)
    estimate_once              — one Monte Carlo batch -> Estimate
    estimate_until_converged    — loop estimate_once until the CI is tight enough
    report / save_estimate      — printing and persistence

estimate_once and estimate_until_converged are circuit_set-specific thin
wrappers (via `_circuit_sampler`) around the two functions the estimation
logic actually lives in, estimate_once_from_sampler and
estimate_until_converged_from_sampler. Those take any `sampler(batch_size,
*, device, dtype, generator) -> Tensor[batch_size, d, d]` callable, so any
ensemble that isn't a circuit_set architecture at all -- e.g. two_designs/'s
random-Clifford calibration, which has no continuous parameters and no
PennyLane gates to trace -- gets the exact same accumulation, pooling, and
confidence-interval machinery for free.

GPU: sample_unitaries builds the batch of unitaries via reshape + einsum
axis-contraction rather than the naive kron(I, gate, I) + matmul
approach — the latter is O(batch * d^3) per gate, this is O(batch * d^2),
which is the best achievable for materialising a full d x d unitary (as
opposed to propagating a single state vector, which is O(batch * d)).
Consecutive single-qubit gates on the same wire are also fused into one
matrix before touching the full tensor, which further cuts the number of
full-tensor passes. Everything is plain torch ops, so it runs on GPU for
free by passing device=torch.device("cuda").
"""

import csv
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import pennylane as qp

from circuits import circuit_set, weight_tensor_shape


# ── device ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _recommended_batch_size_for_d(d: int, device: torch.device,
                                   dtype: torch.dtype = torch.complex64) -> int:
    """Largest N such that the (N, N, d, d) pairwise-trace tensor in
    estimate_once_from_sampler fits comfortably in available memory."""
    bytes_per_element = 8 if dtype == torch.complex64 else 16
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        usable = free_bytes * 0.5
    else:
        usable = 4 * 1024 ** 3  # assume a 4 GB budget when running on CPU
    return max(2, int(math.sqrt(usable / (d * d * bytes_per_element))))


def recommended_batch_size(n_qubits: int, device: torch.device,
                            dtype: torch.dtype = torch.complex64) -> int:
    return _recommended_batch_size_for_d(2 ** n_qubits, device, dtype)


# ── tracing circuit_set(num) into a flat, fused gate list ─────────────────

def _trace_operations(num, n_qubits, weights):
    """Capture circuit_set(num)'s operation sequence via PennyLane's queuing
    mechanism. `weights` carries a trailing batch dimension (shape from
    weight_tensor_shape(...) + (batch_size,)) — circuits.py's circuits index
    into it as params[i, q, ...], so every sliced-out gate parameter comes
    out already batch-shaped, with no per-circuit code needed to support it.
    """
    with qp.queuing.AnnotatedQueue() as q:
        circuit_set(num=num)(weights, wires=list(range(n_qubits)))
    return qp.tape.QuantumScript.from_queue(q).operations


def _fused_gate_list(operations, dtype, device):
    """Merge consecutive single-qubit gates on the same wire into one
    matrix (cheap, O(batch) per merge) so the full-tensor contraction in
    _apply_gate runs once per fused block instead of once per raw gate."""
    pending = {}
    fused = []

    def flush(wire):
        if wire in pending:
            fused.append(((wire,), pending.pop(wire)))

    for op in operations:
        wires = op.wires.tolist()
        G = op.matrix()
        if not torch.is_tensor(G):
            G = torch.as_tensor(G)
        G = G.to(dtype=dtype, device=device)
        if len(wires) == 1:
            w = wires[0]
            pending[w] = G if w not in pending else G @ pending[w]
        elif len(wires) == 2:
            flush(wires[0])
            flush(wires[1])
            fused.append((tuple(wires), G))
        else:
            raise ValueError(
                f"gate {op.name} acts on {len(wires)} wires; "
                "only 1- and 2-qubit gates are supported"
            )
    for w in list(pending):
        flush(w)
    return fused


# ── batched unitary contraction ────────────────────────────────────────

def _apply_gate(U: torch.Tensor, G: torch.Tensor, wires: tuple, n_qubits: int) -> torch.Tensor:
    """Apply gate matrix G (2^k, 2^k) or (batch, 2^k, 2^k) to `wires` of the
    batched unitary U (batch, d, d), in place of a full kron(I, G, I) + matmul.
    Reshapes U's row-index into one axis per qubit, moves the target axes to
    the front, contracts, and moves them back — O(batch * d^2) instead of
    O(batch * d^3) per gate, since only the touched axes are ever expanded.
    """
    B, d, _ = U.shape
    k = len(wires)
    U = U.reshape(B, *([2] * n_qubits), d)
    axes = [1 + w for w in wires]
    dest = list(range(1, 1 + k))
    U = torch.movedim(U, axes, dest)
    rest_shape = U.shape[1 + k:]
    U = U.reshape(B, 2 ** k, -1)
    U = (G @ U) if G.dim() == 2 else torch.einsum("bij,bjk->bik", G, U)
    U = U.reshape(B, *([2] * k), *rest_shape)
    U = torch.movedim(U, dest, axes)
    return U.reshape(B, d, d)


def sample_unitaries(num: int, n_qubits: int, reps: int, batch_size: int, *,
                      device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.complex64,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """batch_size independent samples of circuit_set(num)'s unitary at
    (n_qubits, reps), each with parameters drawn uniformly on [0, 2*pi).
    Returns a (batch_size, d, d) tensor on `device`.

    Verified exact (vs qml.matrix()) for circuits 1-19 and 31-32. Circuit 30
    (qp.StronglyEntanglingLayers, a PennyLane built-in template rather than
    circuits.py's own code) is NOT supported: it validates its weights
    tensor's shape assuming the batch dimension would be leading, not the
    trailing one circuits.py's own circuits are indifferent to, and raises
    a ValueError. Not one of the 19 paper architectures, so left as a
    documented gap rather than special-cased.
    """
    if device is None:
        device = get_device()
    shape = weight_tensor_shape(num, n_qubits, reps)
    weights = 2 * torch.pi * torch.rand(shape + (batch_size,), dtype=torch.float64, generator=generator)
    weights = weights.to(device)

    operations = _trace_operations(num, n_qubits, weights)
    fused = _fused_gate_list(operations, dtype, device)

    d = 2 ** n_qubits
    U = torch.eye(d, dtype=dtype, device=device).expand(batch_size, d, d).clone()
    for wires, G in fused:
        U = _apply_gate(U, G, wires, n_qubits)
    return U


# ── frame potential estimator ──────────────────────────────────────────

@dataclass(frozen=True)
class Estimate:
    """One (possibly pooled) Monte Carlo estimate of F^(t).

    `total`/`sum_sq` are the raw Σ|Tr(Ui†Uj)|^(2t) / Σ|Tr(Ui†Uj)|^(4t) sums
    over n_pairs pairs — additive across independent batches. `variance` is
    the *unbiased* variance of `frame_potential` for this batch: treating
    the n_pairs pairs as independent (naive Var = sum_sq/n_pairs - F^2)
    ignores the row/column correlations from reusing each Ui against every
    Vj, which is negligible at small N but not always (code review §1.4) —
    __add__ pools it correctly across batches (as an n_k-weighted average of
    independent-batch variances) rather than re-deriving a fresh naive one
    from the pooled sums.
    """
    total: float
    sum_sq: float
    variance: float
    n_pairs: int
    t: int
    d: int

    @property
    def frame_potential(self) -> float:
        return self.total / self.n_pairs

    @property
    def haar(self) -> float:
        return float(math.factorial(self.t))

    @property
    def delta(self) -> float:
        return self.frame_potential - self.haar

    @property
    def ratio(self) -> float:
        return self.frame_potential / self.haar

    @property
    def fidelity_error(self) -> float:
        return 1.96 * math.sqrt(max(self.variance, 0.0))

    @property
    def variance_inflation(self) -> float:
        """How much wider the corrected CI is than the naive (independent-
        pairs) one. Near 1 means pair correlations barely matter at this N."""
        naive_variance = max(self.sum_sq / self.n_pairs - self.frame_potential ** 2, 0.0)
        if naive_variance <= 0:
            return float("nan")
        return math.sqrt(self.variance / naive_variance)

    def __add__(self, other: "Estimate") -> "Estimate":
        if self.t != other.t or self.d != other.d:
            raise ValueError("cannot pool Estimates for different t or d")
        n_pairs = self.n_pairs + other.n_pairs
        pooled_variance = (
            self.n_pairs ** 2 * self.variance + other.n_pairs ** 2 * other.variance
        ) / (n_pairs ** 2)
        return Estimate(
            total=self.total + other.total,
            sum_sq=self.sum_sq + other.sum_sq,
            variance=pooled_variance,
            n_pairs=n_pairs,
            t=self.t,
            d=self.d,
        )


def _estimate_from_batches(UA: torch.Tensor, UB: torch.Tensor, t: int, d: int) -> Estimate:
    """Shared math for both estimate_once_from_sampler and the exact
    (whole-group) path: given two independent batches of unitaries, build the
    Estimate from all n_a * n_b cross pairs. UA, UB: (n_a, d, d) / (n_b, d, d)."""
    accum_dtype = torch.float64
    A = UA.unsqueeze(1)
    B = UB.unsqueeze(0)
    traces = torch.einsum("bipq,bjpq->bij", A.conj(), B).squeeze(1)  # (n_a, n_b)
    P = (torch.abs(traces) ** (2 * t)).to(accum_dtype)

    n_a, n_b = P.shape
    total = P.sum().item()
    sum_sq = (torch.abs(traces) ** (4 * t)).to(accum_dtype).sum().item()

    # Unbiased variance for this balanced two-way random-effects layout
    # see Estimate's docstring.
    gm = P.mean()
    r = P.mean(dim=1)
    c = P.mean(dim=0)
    MSA = n_b * ((r - gm) ** 2).sum() / max(n_a - 1, 1)
    MSB = n_a * ((c - gm) ** 2).sum() / max(n_b - 1, 1)
    MSE = ((P - r[:, None] - c[None, :] + gm) ** 2).sum() / max((n_a - 1) * (n_b - 1), 1)
    variance = max(((MSA + MSB - MSE) / (n_a * n_b)).item(), 0.0)

    return Estimate(total=total, sum_sq=sum_sq, variance=variance,
                     n_pairs=n_a * n_b, t=t, d=d)


# A `sampler` is any callable (batch_size, *, device, dtype, generator) ->
# Tensor[batch_size, d, d] of unitaries — sample_unitaries(num, n_qubits, reps,
# ...) partially applied is one, but so is e.g.
# two_designs.family_a_clifford.sample_clifford_unitaries, or any other
# ensemble that isn't a circuit_set architecture at all. Everything below
# this line only depends on the sampler through that interface.

Sampler = Callable[..., torch.Tensor]


def estimate_once_from_sampler(sampler: Sampler, d: int, t: int, n_samples: int, *,
                                device: Optional[torch.device] = None,
                                dtype: torch.dtype = torch.complex64,
                                generator: Optional[torch.Generator] = None) -> Estimate:
    """Draw n_samples unitaries from `sampler` (split into two independent
    halves A, B) and estimate F^(t) from all n_a * n_b cross pairs. `d` is
    the Hilbert space dimension the sampler produces (needed for
    Estimate.d / Estimate.haar, not inferrable from the sampler itself)."""
    if device is None:
        device = get_device()
    n_a = n_samples // 2
    n_b = n_samples - n_a
    UA = sampler(n_a, device=device, dtype=dtype, generator=generator)
    UB = sampler(n_b, device=device, dtype=dtype, generator=generator)
    return _estimate_from_batches(UA, UB, t, d)


def estimate_until_converged_from_sampler(sampler: Sampler, d: int, t: int, *,
                                           n_samples: Optional[int] = None,
                                           rel_tol: float = 0.4,
                                           max_batches: int = 50,
                                           min_abs_error: float = 1e-5,
                                           device: Optional[torch.device] = None,
                                           dtype: torch.dtype = torch.complex64,
                                           generator: Optional[torch.Generator] = None,
                                           verbose: bool = False) -> Estimate:
    """Keep pooling fresh batches (doubling n_samples each time, capped by
    available memory) until the 95% CI is within rel_tol of |delta|, or
    max_batches is reached."""
    if device is None:
        device = get_device()
    if n_samples is None:
        n_samples = d * t * 10  # heuristic starting point (matches d = 2**n_qubits for circuit ensembles)

    max_batch_size = _recommended_batch_size_for_d(d, device, dtype)
    est = estimate_once_from_sampler(sampler, d, t, n_samples,
                                      device=device, dtype=dtype, generator=generator)

    for i in range(max_batches):
        target = abs(rel_tol * est.delta)
        if est.fidelity_error <= target or est.fidelity_error <= min_abs_error:
            break
        if verbose:
            print(f"  batch {i}: F={est.frame_potential:.4f} error={est.fidelity_error:.4f} "
                  f"target={target:.4f} n_pairs={est.n_pairs}")
        n_samples = min(n_samples * 2, max_batch_size)
        est = est + estimate_once_from_sampler(sampler, d, t, n_samples,
                                                device=device, dtype=dtype, generator=generator)

    return est


def _circuit_sampler(num: int, n_qubits: int, reps: int) -> Sampler:
    """The sampler backing every circuit_set architecture: partially applies
    sample_unitaries so it matches the generic (batch_size, *, device, dtype,
    generator) -> Tensor interface."""
    def sampler(batch_size, *, device=None, dtype=torch.complex64, generator=None):
        return sample_unitaries(num, n_qubits, reps, batch_size,
                                 device=device, dtype=dtype, generator=generator)
    return sampler


def estimate_once(num: int, n_qubits: int, reps: int, t: int, n_samples: int, *,
                   device: Optional[torch.device] = None,
                   dtype: torch.dtype = torch.complex64,
                   generator: Optional[torch.Generator] = None) -> Estimate:
    """Draw n_samples unitaries (split into two independent halves A, B) and
    estimate F^(t) from all n_a * n_b cross pairs. Thin circuit_set-specific
    wrapper around estimate_once_from_sampler — use that directly for
    ensembles that aren't a circuit_set architecture (see two_designs/)."""
    return estimate_once_from_sampler(
        _circuit_sampler(num, n_qubits, reps), 2 ** n_qubits, t, n_samples,
        device=device, dtype=dtype, generator=generator,
    )


def estimate_until_converged(num: int, n_qubits: int, reps: int, t: int, *,
                              n_samples: Optional[int] = None,
                              rel_tol: float = 0.4,
                              max_batches: int = 50,
                              min_abs_error: float = 1e-5,
                              device: Optional[torch.device] = None,
                              dtype: torch.dtype = torch.complex64,
                              generator: Optional[torch.Generator] = None,
                              verbose: bool = False) -> Estimate:
    """Keep pooling fresh batches (doubling n_samples each time, capped by
    available memory) until the 95% CI is within rel_tol of |delta|, or
    max_batches is reached. Thin circuit_set-specific wrapper around
    estimate_until_converged_from_sampler."""
    return estimate_until_converged_from_sampler(
        _circuit_sampler(num, n_qubits, reps), 2 ** n_qubits, t,
        n_samples=n_samples, rel_tol=rel_tol, max_batches=max_batches,
        min_abs_error=min_abs_error, device=device, dtype=dtype,
        generator=generator, verbose=verbose,
    )

    return est


def report(est: Estimate, *, circuit_num: int = None, n_qubits: int = None) -> str:
    header = f"circuit {circuit_num}, n_qubits={n_qubits}" if circuit_num is not None else ""
    return (
        f"{header}\n"
        f"  F^({est.t}) (ansatz) : {est.frame_potential:.6f}\n"
        f"  F^({est.t}) (Haar)   : {est.haar:.6f}\n"
        f"  delta F              : {est.delta:.6f}\n"
        f"  ratio F/F_Haar       : {est.ratio:.4f}\n"
        f"  fidelity error       : {est.fidelity_error:.6f}\n"
        f"  variance inflation   : {est.variance_inflation:.3f}\n"
        f"  n_pairs              : {est.n_pairs}"
    )


# ── saving results: one CSV, one row per run ───────────────────────────

FRAME_POTENTIAL_CSV = "results/frame_potential.csv"
FRAME_POTENTIAL_FIELDS = [
    "id", "timestamp", "git_commit",
    "circuit_num", "n_qubits", "reps", "t",
    "frame_potential", "haar_value", "delta", "ratio",
    "variance", "fidelity_error", "variance_inflation",
    "n_pairs", "device", "dtype", "seed", "notes",
]


def _git_commit_hash() -> str:
    """So a saved row can always be traced back to the code that produced
    it (code review §1.5) — "unknown" outside a git repo or with no commits."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_estimate(est: Estimate, *, circuit_num: int, n_qubits: int, reps: int,
                   device, dtype, seed: Optional[int] = None, notes: str = "",
                   path: str = FRAME_POTENTIAL_CSV) -> str:
    """Append one row to `path` (creating it with a header if needed).
    Deriving frame_potential/delta/ratio/fidelity_error from Estimate's
    properties rather than storing them independently makes the "file
    doesn't match what the code would compute" bug from code review §1.5
    structurally impossible here."""
    row_id = str(uuid.uuid4())
    row = {
        "id": row_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "circuit_num": circuit_num,
        "n_qubits": n_qubits,
        "reps": reps,
        "t": est.t,
        "frame_potential": est.frame_potential,
        "haar_value": est.haar,
        "delta": est.delta,
        "ratio": est.ratio,
        "variance": est.variance,
        "fidelity_error": est.fidelity_error,
        "variance_inflation": est.variance_inflation,
        "n_pairs": est.n_pairs,
        "device": str(device),
        "dtype": str(dtype),
        "seed": seed if seed is not None else "",
        "notes": notes,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_POTENTIAL_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row_id


def load_frame_potential(path: str = FRAME_POTENTIAL_CSV):
    """All recorded runs as a pandas DataFrame, ready for groupby/plot."""
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame(columns=FRAME_POTENTIAL_FIELDS)
    return pd.read_csv(path)
