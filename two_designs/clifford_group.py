"""Random Clifford circuits ("Family A" in the "Building 2-Designs" design
notes), used purely as a calibration ensemble.

The n-qubit Clifford group is an exact 3-design (Zhu 2017 / Webb 2016; see
also the "Six families" note this module implements), and fails at t=4
*upward* as it must (frame potential is bounded below by t! for any
ensemble). So F^(t) must come out to exactly t! for t = 1, 2, 3, at every n
-- if your estimator ever returns something below t!, that is a bug in the
estimator, not a discovery. This module exists to make that check cheap.

Two ways to check it, both plugged into frame_potential.py's generic
(sampler -> Estimate) machinery rather than circuit_set (there are no
continuous parameters here, and no PennyLane gates to trace):

  - sample_clifford_unitaries: uniformly random Cliffords (stim's
    Bravyi-Maslov canonical form, O(n^2) per sample) -> Monte Carlo estimate
    via frame_potential.estimate_once_from_sampler /
    estimate_until_converged_from_sampler, same as any circuit_set ensemble.

  - iter_all_clifford_unitaries + exact_estimate_from_group: for n small
    enough to enumerate the *entire* group (n=1: 24 elements, n=2: 11,520),
    F^(t) can be computed exactly -- zero Monte Carlo variance, the
    strongest possible check on the estimator.

Note on global phase: stim.Tableau.to_unitary_matrix's docstring warns the
result's global phase is arbitrary (it fixes it via an internal random state
vector, not a canonical convention). This is harmless here: frame potential
only ever uses |Tr(U^dagger V)|, whose value is invariant to any per-sample
global phase on U or V individually.

Note on reproducibility: stim.Tableau.random exposes no seed/RNG hook (as of
stim 1.16), so the `generator` argument accepted below for interface
compatibility with frame_potential's other samplers is NOT honored -- runs
are not reproducible via `generator`. This is fine for what this module is
for (Monte Carlo estimates with reported confidence intervals, or exact
whole-group sums with no randomness at all); it would matter if you needed
bit-for-bit reproducible Clifford samples elsewhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import stim
import torch

from frame_potential import Estimate


def clifford_group_order(n_qubits: int) -> int:
    """|Cl(n)| modulo global phase (i.e. as tableaus / automorphisms of the
    Pauli group, which is what stim represents and what to_unitary_matrix
    returns one representative phase of). Aaronson & Gottesman 2004, Eq. after
    Thm 1: 2^(n^2+2n) * prod_{j=1}^{n} (4^j - 1). Matches stim.Tableau.iter_all's
    count exactly (checked for n=1: 24, n=2: 11520)."""
    order = 2 ** (n_qubits ** 2 + 2 * n_qubits)
    for j in range(1, n_qubits + 1):
        order *= (4 ** j - 1)
    return order


def sample_clifford_unitaries(n_qubits: int, batch_size: int, *,
                               device: Optional[torch.device] = None,
                               dtype: torch.dtype = torch.complex64,
                               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """batch_size independent uniformly-random n-qubit Clifford unitaries, as
    a (batch_size, d, d) tensor on `device`. Matches the generic
    (batch_size, *, device, dtype, generator) -> Tensor sampler interface
    frame_potential.estimate_once_from_sampler expects.

    `generator` is accepted for interface compatibility only -- see the
    module docstring; it has no effect (stim seeds itself internally).
    """
    del generator  # not honored -- see module docstring
    mats = [stim.Tableau.random(n_qubits).to_unitary_matrix(endian="big")
            for _ in range(batch_size)]
    U = torch.from_numpy(np.stack(mats))  # (batch, d, d), complex64
    return U.to(device=device, dtype=dtype)


def iter_all_clifford_unitaries(n_qubits: int, *,
                                 device: Optional[torch.device] = None,
                                 dtype: torch.dtype = torch.complex64,
                                 max_order: int = 200_000) -> torch.Tensor:
    """*Every* n-qubit Clifford unitary, as a (|Cl(n)|, d, d) tensor -- for
    exact (zero-variance) frame-potential checks via exact_estimate_from_group.
    Only sensible for small n: |Cl(1)| = 24, |Cl(2)| = 11520, |Cl(3)| ~= 9.2e7.
    Raises ValueError above `max_order` rather than silently trying to
    materialize an enormous tensor -- use sample_clifford_unitaries (Monte
    Carlo) for n >= 3.
    """
    order = clifford_group_order(n_qubits)
    if order > max_order:
        raise ValueError(
            f"|Cl({n_qubits})| = {order:,} exceeds max_order={max_order:,}; "
            "exhaustive enumeration is only practical for n=1 (24) or n=2 "
            "(11,520). Use sample_clifford_unitaries for larger n."
        )
    mats = [t.to_unitary_matrix(endian="big") for t in stim.Tableau.iter_all(n_qubits)]
    U = torch.from_numpy(np.stack(mats))
    return U.to(device=device, dtype=dtype)


def exact_estimate_from_group(unitaries: torch.Tensor, t: int, *, row_chunk: int = 2000) -> Estimate:
    """Exact (zero-variance) F^(t) = (1/N^2) sum_{i,j=1}^{N} |Tr(Ui^dag Uj)|^(2t)
    for the ensemble uniform over the N unitaries given (e.g. every element
    of a finite group, from iter_all_clifford_unitaries). This is the literal
    definition of F^(t) for U, V drawn i.i.d. from a *finite* uniform
    ensemble -- including the N "diagonal" i=j pairs, which belong in the sum
    here precisely because this is the whole population, not a sample of it
    (contrast frame_potential.estimate_once_from_sampler, which splits into
    two independent batches specifically so it never needs a diagonal term to
    estimate the same quantity from a *sample*).

    Chunked over rows so the full N x N trace matrix (1.3e8 complex entries,
    ~1 GB, at N=11520) is never fully materialized at once.
    """
    N, d, _ = unitaries.shape
    accum_dtype = torch.float64
    Uc = unitaries.conj()
    total = torch.zeros((), dtype=accum_dtype)
    sum_sq = torch.zeros((), dtype=accum_dtype)
    for start in range(0, N, row_chunk):
        block = Uc[start:start + row_chunk]  # (b, d, d)
        traces = torch.einsum("bij,kij->bk", block, unitaries)  # (b, N)
        p = torch.abs(traces) ** (2 * t)
        total += p.sum().to(accum_dtype)
        sum_sq += (torch.abs(traces) ** (4 * t)).sum().to(accum_dtype)
    return Estimate(total=total.item(), sum_sq=sum_sq.item(), variance=0.0,
                     n_pairs=N * N, t=t, d=d)
