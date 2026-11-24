"""
Frame potential estimation: an Estimate accumulator and the estimators
built on it. The batched-unitary-construction half of this module's
original design now lives in simulation.py (code review, "Three modules
in one" -- simulation, estimation here, I/O in results_io.py);
get_device/apply_embedded_gate/sample_unitaries/recommended_batch_size are
re-exported below so existing `frame_potential.X` call sites (run.py,
extras/benchmark.py, extras/expressibility.py, extras/range_connectivity.py)
keep working unchanged.

Design : a frozen `Estimate` dataclass that behaves like an
accumulator (`estimate_a + estimate_b` pools two independent Monte Carlo
batches), and four small functions :
    sample_unitaries          — batched unitary construction (GPU-friendly, simulation.py)
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
"""

import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import torch

import results_io
from architectures import Architecture
from simulation import (  # noqa: F401 -- re-exported for existing frame_potential.X call sites
    get_device,
    recommended_batch_size,
    apply_embedded_gate,
    sample_unitaries,
    _recommended_batch_size_for_d,
)


# ── frame potential estimator ──────────────────────────────────────────

@dataclass(frozen=True)
class Estimate:
    """One (possibly pooled) Monte Carlo estimate of F^(t).

        'total'/'sum_sq' are the raw Σ|Tr(Ui†Uj)|^(2t) / Σ|Tr(Ui†Uj)|^(4t) sums
        over n_pairs pairs — additive across independent batches. 
        'variance' is the *unbiased* variance of `frame_potential`
        'n_pairs' is the number of independent pairs contributing to this estimate.
        't'/'d' are the frame potential order and Hilbert space dimension.

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
    """given two independent batches of unitaries, 
    build the Estimate from all n_a * n_b cross pairs. UA, UB: (n_a, d, d) / (n_b, d, d).
    """
    accum_dtype = torch.float64
    n_a, n_b = UA.shape[0], UB.shape[0]
    traces = UA.reshape(n_a, -1).conj() @ UB.reshape(n_b, -1).T  # (n_a, n_b)
    P = (torch.abs(traces) ** (2 * t)).to(accum_dtype)

    total = P.sum().item()
    sum_sq = (torch.abs(traces) ** (4 * t)).to(accum_dtype).sum().item()

    # Unbiased variance for this balanced two-way random-effects layout
    gm = P.mean()
    r = P.mean(dim=1)
    c = P.mean(dim=0)
    MSA = n_b * ((r - gm) ** 2).sum() / max(n_a - 1, 1)
    MSB = n_a * ((c - gm) ** 2).sum() / max(n_b - 1, 1)
    MSE = ((P - r[:, None] - c[None, :] + gm) ** 2).sum() / max((n_a - 1) * (n_b - 1), 1)
    variance = max(((MSA + MSB - MSE) / (n_a * n_b)).item(), 0.0)

    return Estimate(total=total, sum_sq=sum_sq, variance=variance,
                     n_pairs=n_a * n_b, t=t, d=d)


Sampler = Callable[..., torch.Tensor]
"""
A `Sampler` is any callable (batch_size, *, device, dtype, generator) ->
Tensor[batch_size, d, d] of unitaries — sample_unitaries(num, n_qubits, reps,
...) partially applied is one, but so is a function that generates random unitaries.
"""

def _is_oom(exc: RuntimeError) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _estimate_once_with_batch_size(sampler: Sampler, d: int, t: int, n_samples: int, *,
                                    device: Optional[torch.device] = None,
                                    dtype: torch.dtype = torch.complex64,
                                    generator: Optional[torch.Generator] = None):
    """Same backoff behavior as estimate_once_from_sampler, but also returns
    the batch size `n` that actually succeeded -- exactly, not re-derived
    from Estimate.n_pairs via round(2*sqrt(n_pairs)) (an approximation,
    only exact for even n). estimate_until_converged_from_sampler needs
    this to clamp its own doubling after a backoff; every other caller
    goes through estimate_once_from_sampler below, which just discards it.
    """
    if device is None:
        device = get_device()
    n = n_samples
    while True:
        n_a = n // 2
        n_b = n - n_a
        try:
            UA = sampler(n_a, device=device, dtype=dtype, generator=generator)
            UB = sampler(n_b, device=device, dtype=dtype, generator=generator)
            return _estimate_from_batches(UA, UB, t, d), n
        except RuntimeError as e:
            if not _is_oom(e) or n <= 4:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            n_next = max(4, n // 2)
            print(f"  [frame_potential] OOM at n_samples={n} (device={device}) -- "
                  f"retrying at {n_next}")
            n = n_next


def estimate_once_from_sampler(sampler: Sampler, d: int, t: int, n_samples: int, *,
                                device: Optional[torch.device] = None,
                                dtype: torch.dtype = torch.complex64,
                                generator: Optional[torch.Generator] = None) -> Estimate:
    """Draw n_samples unitaries from `sampler` (split into two independent
    halves A, B) and estimate F^(t) from all n_a * n_b cross pairs.
    `d` is the Hilbert space dimension the sampler produces (needed for
    Estimate.d / Estimate.haar, not inferrable from the sampler itself).

    `_recommended_batch_size_for_d` (what callers use to pick n_samples) only
    estimates _estimate_from_batches's own peak memory -- it has no way to
    know the sampler's peak, which for sample_unitaries includes whatever
    apply_embedded_gate's reshape/movedim pipeline needs transiently per gate,
    on top of the (batch, d, d) unitary itself. That gap is what actually
    OOMs in practice at large d, not the formula being wrong about the part
    it does model. Rather than pad the formula with a guessed safety factor
    for a cost it can't see, this backs off and retries on an actual CUDA
    OOM: halve n_samples, free the cache, try again, down to a floor of 4.
    The returned Estimate.n_pairs honestly reflects whatever batch size
    actually succeeded, so callers (including estimate_until_converged_from_sampler,
    which just sees a smaller-than-requested batch and keeps looping) don't
    need to know this happened.
    """
    est, _ = _estimate_once_with_batch_size(sampler, d, t, n_samples, device=device,
                                             dtype=dtype, generator=generator)
    return est


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
        n_samples = d * t * 10  # heuristic starting point

    max_batch_size = _recommended_batch_size_for_d(d, device, dtype)
    n_samples = min(n_samples, max_batch_size)  # the heuristic above grows with d; the memory
                                                 # cap shrinks with d -- clamp before the first
                                                 # call too, not just the doublings below, or a
                                                 # large-d first call can ask for far more than
                                                 # fits (this was the actual OOM cause -- see
                                                 # _estimate_from_batches's docstring for the rest)
    est, achieved = _estimate_once_with_batch_size(sampler, d, t, n_samples,
                                                    device=device, dtype=dtype, generator=generator)
    # estimate_once backs off silently on OOM, so what it actually used can
    # be smaller than what was asked for. If it backed off, treat that as
    # the new ceiling instead of max_batch_size, or every doubling below
    # would re-discover the exact same OOM (and pay its own backoff sub-loop)
    # from scratch every single iteration.
    if achieved < n_samples:
        max_batch_size = min(max_batch_size, achieved)
        n_samples = achieved

    for i in range(max_batches):
        target = abs(rel_tol * est.delta)
        if est.fidelity_error <= target or est.fidelity_error <= min_abs_error:
            break
        if verbose:
            print(f"  batch {i}: F={est.frame_potential:.4f} error={est.fidelity_error:.4f} "
                  f"target={target:.4f} n_pairs={est.n_pairs}")
        n_samples = min(n_samples * 2, max_batch_size)
        new_est, achieved = _estimate_once_with_batch_size(sampler, d, t, n_samples,
                                                             device=device, dtype=dtype,
                                                             generator=generator)
        if achieved < n_samples:
            max_batch_size = min(max_batch_size, achieved)
            n_samples = achieved
        est = est + new_est

    return est


def _circuit_sampler(arch: Architecture, n_qubits: int, reps: int) -> Sampler:
    """The sampler backing every circuit_set/generic_ansatz architecture:
    partially applies sample_unitaries so it matches the generic
    (batch_size, *, device, dtype, generator) -> Tensor interface."""
    def sampler(batch_size, *, device=None, dtype=torch.complex64, generator=None):
        return sample_unitaries(arch, n_qubits, reps, batch_size,
                                 device=device, dtype=dtype, generator=generator)
    return sampler


def estimate_once(arch: Architecture, n_qubits: int, reps: int, t: int, n_samples: int, *,
                   device: Optional[torch.device] = None,
                   dtype: torch.dtype = torch.complex64,
                   generator: Optional[torch.Generator] = None) -> Estimate:
    """
    estimate_once_from_sampler Wrapper for circuit_set/generic_ansatz architectures.
    """
    return estimate_once_from_sampler(
        _circuit_sampler(arch, n_qubits, reps), 2 ** n_qubits, t, n_samples,
        device=device, dtype=dtype, generator=generator,
    )


def estimate_until_converged(arch: Architecture, n_qubits: int, reps: int, t: int, *,
                              n_samples: Optional[int] = None,
                              rel_tol: float = 0.4,
                              max_batches: int = 50,
                              min_abs_error: float = 1e-5,
                              device: Optional[torch.device] = None,
                              dtype: torch.dtype = torch.complex64,
                              generator: Optional[torch.Generator] = None,
                              verbose: bool = False) -> Estimate:
    """
        estimate_until_converged Wrapper for circuit_set/generic_ansatz architectures.
        """
    return estimate_until_converged_from_sampler(
        _circuit_sampler(arch, n_qubits, reps), 2 ** n_qubits, t,
        n_samples=n_samples, rel_tol=rel_tol, max_batches=max_batches,
        min_abs_error=min_abs_error, device=device, dtype=dtype,
        generator=generator, verbose=verbose,
    )



def report(est: Estimate, *, arch: Architecture = None, n_qubits: int = None, reps: int = None) -> str:
    header = f"{arch.label}, n_qubits={n_qubits}, reps={reps}" if arch is not None else ""
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
    "circuit_num", "generic", "n_qubits", "reps", "n_params", "t",
    "frame_potential", "haar_value", "delta", "ratio",
    "variance", "fidelity_error", "variance_inflation",
    "n_pairs", "device", "dtype", "seed", "notes",
]


def save_estimate(est: Estimate, *, arch: Architecture, n_qubits: int, reps: int,
                   device, dtype, seed: Optional[int] = None, notes: str = "",
                   path: str = FRAME_POTENTIAL_CSV) -> str:
    """Append one row to `path` (creating it with a header if needed).
    Deriving frame_potential/delta/ratio/fidelity_error from Estimate's
    properties."""
    row_id = str(uuid.uuid4())
    row = {
        "id": row_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": results_io.git_commit_hash(),
        "circuit_num": arch.num,
        "generic": arch.generic,
        "n_qubits": n_qubits,
        "reps": reps,
        "n_params": arch.n_trainable(n_qubits, reps),
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
    results_io.append_row(path, FRAME_POTENTIAL_FIELDS, row)
    return row_id


def load_frame_potential(path: str = FRAME_POTENTIAL_CSV):
    """All recorded runs as a pandas DataFrame, ready for groupby/plot."""
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame(columns=FRAME_POTENTIAL_FIELDS)
    return pd.read_csv(path)
