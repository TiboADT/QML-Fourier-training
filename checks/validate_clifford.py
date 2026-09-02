"""Calibration check for the frame-potential estimator, using random
Clifford circuits ("Family A" in the "Building 2-Designs" design notes).

The Clifford group is an exact 3-design, so this is the one ensemble where
the right answer is known in closed form: F^(t) = t! for t = 1, 2, 3, at
every n. Two independent checks:

  1. Exact, zero-variance: sum over the *entire* Clifford group at n=1
     (24 elements) and n=2 (11,520 elements) via exact_estimate_from_group.
     No sampling anywhere -- if this doesn't hit t! to float precision,
     the bug is in exact_estimate_from_group or in Estimate itself.

  2. Monte Carlo: sample_clifford_unitaries pushed through
     frame_potential's generic estimate_once_from_sampler, at n = 4, 6 --
     the same code path any future two_designs family will use. If the
     reported F^(t) isn't within its own confidence interval of t!, the bug
     is in the *general* Monte Carlo estimator, not in anything specific to
     Cliffords.

     Deliberately estimate_once_from_sampler (a fixed sample count) rather
     than estimate_until_converged_from_sampler here: convergence there is
     judged by rel_tol * |delta|, and delta = F^(t) - t! is supposed to be
     ~0 for an exact design -- so the target the loop chases shrinks along
     with the thing it's measuring, and it will happily burn all
     max_batches doubling the sample size chasing noise. That failure mode
     is a real property of the relative-tolerance stopping rule (it'll bite
     any near-exact-design ensemble, not just this one) rather than
     anything specific to Cliffords -- worth knowing about, not something
     to paper over here.

Run from the repo root: python check.py validate --only clifford
(or the short alias: python check.py validate --only a)
"""

import math
import time

import torch

import frame_potential as fp
from two_designs.clifford_group import (
    clifford_group_order,
    exact_estimate_from_group,
    iter_all_clifford_unitaries,
    sample_clifford_unitaries,
)


def check_exact(n_qubits: int, ts=(1, 2, 3)):
    """t! is only the exact Haar value of F^(t) for t <= d (Prop. 1/3 in the
    design notes -- Schur-Weyl needs the t! permutation operators to stay
    linearly independent, which fails past t = d). t values above d = 2**n_qubits
    are skipped rather than flagged, not silently passed."""
    d = 2 ** n_qubits
    order = clifford_group_order(n_qubits)
    print(f"--- exact, n_qubits={n_qubits} (|Cl({n_qubits})| = {order:,}, d={d}) ---")
    t0 = time.time()
    U = iter_all_clifford_unitaries(n_qubits, dtype=torch.complex128)
    print(f"  enumerated {U.shape[0]} unitaries in {time.time() - t0:.1f}s")
    for t in ts:
        if t > d:
            print(f"  t={t}  skipped (t > d={d}: t! is not the exact Haar value here)")
            continue
        est = exact_estimate_from_group(U, t)
        haar = math.factorial(t)
        # stim's to_unitary_matrix is natively complex64 (cast up to complex128
        # for accumulation), so ~1e-7-level residuals are float32 rounding, not
        # a real mismatch -- 1e-4 comfortably clears that while still catching
        # any real bug, which would show up at the 10-50% level, not 1e-4.
        ok = "OK" if abs(est.frame_potential - haar) < 1e-4 else "MISMATCH"
        print(f"  t={t}  F={est.frame_potential:.12f}  Haar={haar}  "
              f"|delta|={abs(est.frame_potential - haar):.2e}  [{ok}]")
    print()


def check_sampled(n_qubits: int, n_samples: int, ts=(1, 2, 3)):
    print(f"--- Monte Carlo, n_qubits={n_qubits}, n_samples={n_samples} ---")

    def sampler(batch_size, *, device=None, dtype=torch.complex64, generator=None):
        return sample_clifford_unitaries(n_qubits, batch_size, device=device,
                                          dtype=dtype, generator=generator)

    d = 2 ** n_qubits
    device = torch.device("cpu")
    for t in ts:
        est = fp.estimate_once_from_sampler(
            sampler, d, t, n_samples, device=device, dtype=torch.complex128,
        )
        haar = est.haar
        within_ci = abs(est.delta) <= max(est.fidelity_error, 1e-9)
        print(f"  t={t}  F={est.frame_potential:.4f}  Haar={haar:.1f}  "
              f"delta={est.delta:+.4f}  95% CI +/-{est.fidelity_error:.4f}  "
              f"[{'within CI' if within_ci else 'outside CI'}]")
    print()


def main():
    check_exact(1)
    check_exact(2)
    check_sampled(4, n_samples=4000)
    check_sampled(6, n_samples=2000)


if __name__ == "__main__":
    main()
