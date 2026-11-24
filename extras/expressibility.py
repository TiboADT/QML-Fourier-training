"""Expressibility (Kullback-Leibler divergence against Haar) -- not used by
the paper, moved here from frame_potential.py (code review, section 5/"must
fix" item "remove the KL block": nothing in run.py/checks/notebooks calls
these). Kept because the math is correct and may be useful later, just not
part of the paper's reproducible path.

Sim, Johnson & Aspuru-Guzik (2019, arXiv:1905.10876) define a circuit's
"expressibility" as D_KL(P_hat || P_Haar) between the empirical
distribution of fidelities F = |<0|U_i^dag U_j|0>|^2 sampled from the
circuit and the exact Haar-random one. F is the squared magnitude of the
(0,0) entry of U_i^dag U_j -- since |0> is the first computational basis
vector, U|0> is just U's first column, so this is the squared overlap of
U_i's and U_j's first columns, not anything read off frame potential's
trace (which sums *all* diagonal entries, not one). Related quantity,
different slice -- reuses sample_unitaries and the exact same batched-
GEMM trick _estimate_from_batches's trace computation uses, just on
(batch, d) column vectors instead of (batch, d^2) flattened matrices.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frame_potential import get_device, sample_unitaries
from architectures import Architecture


def sample_fidelities(arch: Architecture, n_qubits: int, reps: int, batch_size: int, *,
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.complex64,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """All pairwise fidelities F = |<0|U_i^dag U_j|0>|^2 between the two
    independent halves of a batch_size-sample draw of arch's output state
    (starting from |0...0>) -- see this module's docstring for why that's
    just the overlap of U's first columns. Splitting into two independent
    halves (same convention as estimate_once_from_sampler) means no U_i is
    ever compared against itself, which would be an identically-1
    self-overlap biasing the distribution.

    Returns a flat real Tensor[n_a * n_b] of fidelity samples. Unlike
    estimate_once_from_sampler, this does not retry on CUDA OOM by halving
    the batch -- pass a smaller batch_size directly if that happens; kept
    out for this first, deliberately light version.
    """
    if device is None:
        device = get_device()
    n_a = batch_size // 2
    n_b = batch_size - n_a
    UA = sample_unitaries(arch, n_qubits, reps, n_a,
                           device=device, dtype=dtype, generator=generator)
    UB = sample_unitaries(arch, n_qubits, reps, n_b,
                           device=device, dtype=dtype, generator=generator)
    psi_A = UA[:, :, 0]  # (n_a, d): U_i|0> -- U's first column
    psi_B = UB[:, :, 0]  # (n_b, d)
    overlaps = psi_A.conj() @ psi_B.T  # (n_a, n_b)
    return (overlaps.abs() ** 2).reshape(-1).to(torch.float64)


def _haar_fidelity_bin_probs(d: int, n_bins: int) -> torch.Tensor:
    """Exact Haar-random fidelity probability mass in each of n_bins
    equal-width bins over [0, 1], from the closed-form Haar fidelity CDF
    P(F' <= F) = 1 - (1 - F)^(d - 1) (Sim et al. 2019 Eq. 16's P_Haar(F) =
    (d-1)(1-F)^(d-2) is that CDF's derivative). Exact, not estimated by
    sampling -- zero variance on the Haar side of the KL divergence, since
    the closed form is already known; every bin's mass is strictly
    positive for any d >= 2 and any n_bins, so the KL sum below never
    divides by zero on the Haar side.

    Computed as survival(lo) - survival(hi), survival(F) = (1-F)^(d-1) =
    1 - CDF(F), not CDF(hi) - CDF(lo) directly: for the tail bins at
    moderately large d, CDF(F) rounds to a value extremely close to 1
    (since survival(F) itself is tiny there), so subtracting two such
    CDF values loses precision to catastrophic cancellation -- confirmed
    by comparing against the exact closed form (d-1)*ln(n_bins) for the
    degenerate all-fidelity-1 case, which drifted by ~0.001% at d=8 with
    the CDF-difference version and is exact with this one. survival(lo)
    and survival(hi) are each a plain power of a number not close to 1 or
    0 (for any bin not already at the very edge), so this difference
    doesn't have the same cancellation problem.
    """
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    survival = (1 - edges) ** (d - 1)
    return survival[:-1] - survival[1:]


@dataclass(frozen=True)
class KLEstimate:
    """One (possibly pooled) estimate of expressibility: the KL divergence
    between a circuit's empirical fidelity distribution and the exact
    analytic Haar one, both histogrammed into n_bins equal-width bins over
    [0, 1]. Lower is more expressible (closer to Haar-random); the least-
    expressible case (a circuit whose output state never changes) approaches
    (d-1)*ln(n_bins) -- confirmed against Sim et al. 2019's own worked
    example: their single-qubit (d=2) idle circuit gives D_KL = ln(75) =
    4.3175, which they round to 4.30.

    counts: raw per-bin fidelity sample counts. Unlike frame potential's
    Estimate, pooling across batches is exact simple addition (no weighted-
    variance bookkeeping needed) -- a histogram's bin counts from
    independent batches just add.
    """
    counts: torch.Tensor      # (n_bins,) int64
    haar_probs: torch.Tensor  # (n_bins,) float64, sums to 1
    n_bins: int
    d: int

    @property
    def n_samples(self) -> int:
        return int(self.counts.sum().item())

    @property
    def kl_divergence(self) -> float:
        p = self.counts.to(torch.float64) / self.n_samples
        nonzero = p > 0
        return float((p[nonzero] * torch.log(p[nonzero] / self.haar_probs[nonzero])).sum().item())

    def __add__(self, other: "KLEstimate") -> "KLEstimate":
        if self.n_bins != other.n_bins or self.d != other.d:
            raise ValueError("cannot pool KLEstimates for different n_bins or d")
        return KLEstimate(counts=self.counts + other.counts, haar_probs=self.haar_probs,
                           n_bins=self.n_bins, d=self.d)


def expressibility_once(arch: Architecture, n_qubits: int, reps: int, n_samples: int, *,
                         n_bins: int = 75,
                         device: Optional[torch.device] = None,
                         dtype: torch.dtype = torch.complex64,
                         generator: Optional[torch.Generator] = None) -> KLEstimate:
    """One Monte Carlo batch's worth of expressibility (KL divergence vs.
    Haar) for arch. n_bins=75 matches Sim, Johnson &
    Aspuru-Guzik (2019)'s own convention. See sample_fidelities/KLEstimate.
    """
    d = 2 ** n_qubits
    fidelities = sample_fidelities(arch, n_qubits, reps, n_samples,
                                    device=device, dtype=dtype, generator=generator)
    counts = torch.histc(fidelities.cpu(), bins=n_bins, min=0.0, max=1.0)
    haar_probs = _haar_fidelity_bin_probs(d, n_bins)
    return KLEstimate(counts=counts.to(torch.int64), haar_probs=haar_probs, n_bins=n_bins, d=d)
