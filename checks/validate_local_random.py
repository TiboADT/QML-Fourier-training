"""Calibration check for local random circuits ("Family B" in the "Building
2-Designs" design notes): circuit 34 (Haar-reparametrized KAK1 blocks,
brickwork-tiled across N qubits, backed by two_designs/haar_reparam.py) and
its ablation twin, circuit 33 (identical gate structure, raw angles used
directly instead of Haar-reparametrized -- no correction at all).

Unlike the Clifford-group check (checks/validate_clifford.py), there is no
finite group to enumerate here -- SU(4) is a continuous (Lie) group, so
there's no exact, zero-variance check available; every check below is
Monte Carlo.

There IS a second difference from the Clifford group worth being explicit
about: circuit 34 is not a *mathematically exact* design the way the
Clifford group is. Its local dressings are exact closed-form Haar-SU(2)
(see haar_reparam.euler_angles), but its non-local (canonical) angles go
through an *empirically built* Rosenblatt transform (haar_reparam.sample_canonical,
fit from a finite Ginibre-sampled dataset -- see that module's docstring).
So F^(t) for circuit 34 has two sources of deviation from t!:
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
    """Circuit 34 at n_qubits=2 against Haar, fixed sample count -- same
    reasoning as the Clifford-group check's check_sampled: pick a sample
    size that gives a usefully tight CI, don't chase convergence.

    reps=3, not 1: at n_qubits=2 the alternating-offset brickwork's
    layer_pairs alternates [1, 0, 1, 0, ...] (see
    circuits._brickwork_layer_pairs) -- layer 0 is dressing-only by
    construction, and layer 1 is an odd layer that fires zero gates at
    this qubit count, so reps=1 or 2 would test a product of two
    disconnected Haar-SU(2) rotations, not an actual KAK1 block. reps=3
    is the smallest value that actually applies one."""
    print(f"--- circuit 34 (single block, n_qubits=2, reps=3), n_samples={n_samples} ---")
    device = torch.device("cpu")
    for t in ts:
        est = fp.estimate_once(34, n_qubits=2, reps=3, t=t, n_samples=n_samples,
                                device=device, dtype=torch.complex128)
        within_ci = abs(est.delta) <= max(est.fidelity_error, 1e-9)
        print(f"  t={t}  F={est.frame_potential:.4f}  Haar={est.haar:.1f}  "
              f"delta={est.delta:+.4f}  95% CI +/-{est.fidelity_error:.4f}  "
              f"[{'within CI' if within_ci else 'outside CI'}]")
    print()


def check_brickwork_depth(n_qubits=4, reps_list=(1, 2, 4, 8), t=2, n_samples=4000):
    """Circuit 34 (brickwork of Haar-exact KAK1 blocks) should approach
    Haar as reps grows -- it is NOT itself Haar-random for reps=1 on
    n_qubits > 2 (that's the whole point of the local-random-circuit
    ensemble: approximate, depth-dependent)."""
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


def check_convergence_pathology(n_qubits=2, reps=3, t=2, rel_tol=0.3, max_batches=50):
    """Does estimate_until_converged hit max_batches without satisfying its
    own stopping rule, the way it did on the (exactly-designed) Clifford
    ensemble? Circuit 34's delta is small (it's *supposed* to be near-Haar),
    so the same failure mode -- a relative target that shrinks about as fast
    as achievable precision -- is expected here too. This runs it and
    reports what actually happened rather than assuming.

    reps=3, not 1: see check_single_block's docstring -- n_qubits=2 needs
    reps>=3 to apply an actual KAK1 block at all."""
    print(f"--- convergence-loop diagnostic: estimate_until_converged(34, t={t}) ---")
    device = torch.device("cpu")
    t0 = time.time()
    est = fp.estimate_until_converged(34, n_qubits=n_qubits, reps=reps, t=t, rel_tol=rel_tol,
                                       max_batches=max_batches, device=device,
                                       dtype=torch.complex128)
    elapsed = time.time() - t0
    target = abs(rel_tol * est.delta)
    converged = est.fidelity_error <= target or est.fidelity_error <= 1e-5
    within_ci = abs(est.delta) <= max(est.fidelity_error, 1e-9)
    print(f"  t={t}  F={est.frame_potential:.4f}  Haar={est.haar:.1f}  delta={est.delta:+.4f}  "
          f"95% CI +/-{est.fidelity_error:.4f}  "
          f"[{'converged' if converged else 'not converged'}] [{'within CI' if within_ci else 'outside CI'}]")
    print(f"  n_pairs={est.n_pairs:,}  elapsed={elapsed:.1f}s  target={target:.4f}")
    if not converged:
        print("  (expected here: delta is small by construction, so the relative stopping "
              "target shrinks about as fast as achievable precision -- same failure mode as "
              "the Clifford ensemble. Use estimate_once with a fixed sample count instead; "
              "see check_single_block above.)")
    print()


def main(argv=None):
    del argv  # no flags of its own -- accepted for a uniform call signature, see checks/validate.py
    check_single_block()
    check_brickwork_depth()
    check_convergence_pathology()


if __name__ == "__main__":
    main()
