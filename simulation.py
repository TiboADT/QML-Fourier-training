"""Batched unitary construction from circuits.py/generic_ansatz.py circuit
definitions, via architectures.Architecture -- the GPU-friendly half of
frame_potential.py's original design, split out (code review, "Three
modules in one": simulation, estimation, I/O -- I/O is results_io.py,
estimation stays in frame_potential.py, this is simulation).

sample_unitaries builds the batch of unitaries via reshape + axis-
contraction rather than the naive kron(I, gate, I) + matmul approach — the
latter is O(batch * d^3) per gate, this is O(batch * d^2), which is the
best achievable for materialising a full d x d unitary (as opposed to
propagating a single state vector, which is O(batch * d)). Consecutive
single-qubit gates on the same wire are also fused into one matrix before
touching the full tensor, which further cuts the number of full-tensor
passes. Everything is plain torch ops, so it runs on GPU for free by
passing device=torch.device("cuda").
"""

import math
from typing import Optional

import torch
import pennylane as qp

from architectures import Architecture


# ── device ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _recommended_batch_size_for_d(d: int, device: torch.device,
                                   dtype: torch.dtype = torch.complex64) -> int:
    """Largest N such that _estimate_from_batches's peak memory -- two
    (N, d^2) reshaped batches plus the (N, N) GEMM output, peak = 2*N*d^2*bpe
    + N^2*bpe -- fits comfortably in available memory."""

    bytes_per_element = 8 if dtype == torch.complex64 else 16
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        usable = free_bytes * 0.5
    else:
        usable = 4 * 1024 ** 3  # assume a 4 GB budget when running on CPU
    a, b = bytes_per_element, 2 * d * d * bytes_per_element
    n = (-b + math.sqrt(b * b + 4 * a * usable)) / (2 * a)
    return max(2, int(n))


def recommended_batch_size(n_qubits: int, device: torch.device,
                            dtype: torch.dtype = torch.complex64) -> int:
    return _recommended_batch_size_for_d(2 ** n_qubits, device, dtype)


# ── tracing an Architecture into a flat, fused gate list ───────────────

def _trace_operations(arch: Architecture, n_qubits, weights):
    """Capture arch's operation sequence via PennyLane's queuing mechanism.
    `weights` carries a trailing batch dimension (shape from
    arch.weight_shape(...) + (batch_size,)) — circuits.py's circuits index
    into it as params[i, q, ...], so every sliced-out gate parameter comes
    out already batch-shaped, with no per-circuit code needed to support it.
    """
    with qp.queuing.AnnotatedQueue() as q:
        arch.build(weights, wires=list(range(n_qubits)))
    return qp.tape.QuantumScript.from_queue(q).operations


def _fused_gate_list(operations, dtype, device):
    """Merge consecutive single-qubit gates on the same wire into one
    matrix (cheap, O(batch) per merge) """
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

def apply_embedded_gate(U: torch.Tensor, G: torch.Tensor, wires: tuple, n_qubits: int) -> torch.Tensor:
    """Apply gate matrix G (2^k, 2^k) or (batch, 2^k, 2^k) to `wires` of the
    batched unitary U (batch, d, d).
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


def sample_unitaries(arch: Architecture, n_qubits: int, reps: int, batch_size: int, *,
                      device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.complex64,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """batch_size independent samples of arch's unitary at (n_qubits, reps),
    each with parameters drawn uniformly on [0, 2*pi). Returns a
    (batch_size, d, d) tensor on `device`.
    """
    if device is None:
        device = get_device()
    shape = arch.weight_shape(n_qubits, reps)
    weights = 2 * torch.pi * torch.rand(shape + (batch_size,), dtype=torch.float64, generator=generator)
    weights = weights.to(device)

    operations = _trace_operations(arch, n_qubits, weights)
    fused = _fused_gate_list(operations, dtype, device)

    d = 2 ** n_qubits
    U = torch.eye(d, dtype=dtype, device=device).expand(batch_size, d, d).clone()
    for wires, G in fused:
        U = apply_embedded_gate(U, G, wires, n_qubits)
    return U
