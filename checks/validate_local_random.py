"""Calibration check for local random circuits ("Family B" in the "Building
2-Designs" design notes): circuits 33/34, backed by
two_designs/haar_reparam.py. The KAK1 2-qubit block (circuit 33) is
Haar-random by construction (up to the reparametrization's own residual
bias, see below), and a brickwork of them (circuit 34) is the standard
local-random-circuit approximate-design ensemble from the literature.

Unlike the Clifford-group check (checks/validate_clifford.py), there is no
finite group to enumerate here -- SU(4) is a continuous (Lie) group, so
there's no exact, zero-variance check available; every check below is
Monte Carlo.

There IS a second difference from the Clifford group worth being explicit
about: circuit 33 is not a *mathematically exact* design the way the
Clifford group is. Its 12 local angles are exact closed-form Haar-SU(2)
(see haar_reparam.euler_angles), but its 3 non-local angles go through an
*empirically built* Rosenblatt transform (haar_reparam.sample_canonical,
fit from a finite Ginibre-sampled dataset -- see that module's docstring).
So F^(t) for circuit 33 has two sources of deviation from t!:
  (a) genuine Monte Carlo sampling noise (shrinks with more samples), and
  (b) a small systematic bias from the transform's finite table resolution
      (does NOT shrink with more samples -- only with a finer table).
That distinction is exactly why estimate_until_converged is the wrong tool
here (see check_convergence_pathology below): its stopping rule assumes
delta -> 0 is achievable by sampling more, which is only true of (a).

Run from the repo root: python check.py validate --only local-random
(or the short alias: python check.py validate --only b)
"""

import math
import time

import torch

import frame_potential as fp


def check_single_block(n_samples=8000, ts=(1, 2, 3)):
    """Circuit 33 (a single 2-qubit KAK1 block) against Haar, fixed sample
    count -- same reasoning as the Clifford-group check's check_sampled:
    pick a sample size that gives a usefully tight CI, don't chase
    convergence."""
    print(f"--- circuit 33 (single block), n_samples={n_samples} ---")
    device = torch.device("cpu")
    for t in ts:
        est = fp.estimate_once(33, n_qubits=2, reps=1, t=t, n_samples=n_samples,
                                device=device, dtype=torch.complex128)
        within_ci = abs(est.delta) <= max(est.fidelity_error, 1e-9)
        print(f"  t={t}  F={est.frame_potential:.4f}  Haar={est.haar:.1f}  "
              f"delta={est.delta:+.4f}  95% CI +/-{est.fidelity_error:.4f}  "
              f"[{'within CI' if within_ci else 'outside CI'}]")
    print()


def check_brickwork_depth(n_qubits=4, reps_list=(1, 2, 4, 8), t=2, n_samples=4000):
    """Circuit 34 (brickwork of the same block) should approach Haar as
    reps grows -- it is NOT itself Haar-random for reps=1 on n_qubits > 2
    (that's the whole point of the local-random-circuit ensemble:
    approximate, depth-dependent)."""
    print(f"--- circuit 34 (brickwork), n_qubits={n_qubits}, t={t} ---")
    device = torch.device("cpu")
    haar = math.factorial(t)
    for reps in reps_list:
        est = fp.estimate_once(34, n_qubits=n_qubits, reps=reps, t=t, n_samples=n_samples,
                                device=device, dtype=torch.complex128)
        print(f"  reps={reps:<3d}  F={est.frame_potential:.4f}  Haar={haar:.1f}  "
              f"ratio={est.ratio:.4f}")
    print("  (ratio should trend towards 1.0 as reps grows)")
    print()


def check_convergence_pathology(n_qubits=2, t=2, rel_tol=0.3, max_batches=50):
    """Does estimate_until_converged hit max_batches without satisfying its
    own stopping rule, the way it did on the (exactly-designed) Clifford
    ensemble? Circuit 33's delta is small (it's *supposed* to be near-Haar),
    so the same failure mode -- a relative target that shrinks about as fast
    as achievable precision -- is expected here too. This runs it and
    reports what actually happened rather than assuming."""
    print(f"--- convergence-loop diagnostic: estimate_until_converged(33, t={t}) ---")
    device = torch.device("cpu")
    t0 = time.time()
    est = fp.estimate_until_converged(33, n_qubits=n_qubits, reps=1, t=t, rel_tol=rel_tol,
                                       max_batches=max_batches, device=device,
                                       dtype=torch.complex128)
    elapsed = time.time() - t0
    target = abs(rel_tol * est.delta)
    converged = est.fidelity_error <= target or est.fidelity_error <= 1e-5
    print(f"  F={est.frame_potential:.4f}  delta={est.delta:+.4f}  "
          f"fidelity_error={est.fidelity_error:.4f}  target={target:.4f}")
    print(f"  n_pairs={est.n_pairs:,}  elapsed={elapsed:.1f}s")
    if converged:
        print("  -> stopped because its own criterion was satisfied.")
    else:
        print("  -> hit max_batches WITHOUT satisfying its own criterion "
              "(same failure mode as the Clifford ensemble in validate_clifford.py: "
              "delta is small, so the relative target shrinks about as fast as the "
              "achievable precision). Use estimate_once with a fixed sample count "
              "instead for a near-exact-design ensemble like this one -- "
              "see check_single_block above.")
    print()


def main():
    check_single_block()
    check_brickwork_depth()
    check_convergence_pathology()


if __name__ == "__main__":
    main()
