"""Calibration + comparison check for range-limited connectivity
(two_designs/range_connectivity.py) -- "Family C" in the "Building
2-Designs" design notes: same exact-Haar KAK1 block as circuit 34 in the
local-random check, same gate count, but each layer wires up a fresh
random matching restricted to pairs at most `max_range` apart, instead of
a fixed nearest-neighbour brickwork. max_range=1 is "local" in the same
style as circuit 34; max_range >= n_qubits-1 is "fully connected" /
"permuted brickwork" -- no restriction on which pair a gate can touch.

Unlike checks/validate_clifford.py and checks/validate_local_random.py,
the point here isn't just "does this match Haar" -- it's "how does
approach-to-Haar depend on connectivity range, at matched gate count and
depth". So the main thing this script does is print a comparison table
(sweep_gain), the direct answer to "how much do I gain from letting gates
reach further". check_sane runs first as a quick sanity check (unitary,
det=1, roughly Haar at generous depth+range) so a construction bug shows
up as a loud failure rather than a confusing sweep table.

Run from the repo root: python check.py validate --only connectivity
(or the short alias: python check.py validate --only c)

Custom sweep, forwarded through check.py's own argv passthrough:
    python check.py validate --only c -- --n-qubits 8 --reps 2 4 8 16 --ranges 1 2 4 7
"""

import argparse
import math

import torch

import frame_potential as fp
from two_designs.range_connectivity import sample_range_connected_unitaries


def _sampler(n_qubits, reps, max_range, samples_per_wiring):
    def sampler(batch_size, *, device=None, dtype=torch.complex64, generator=None):
        return sample_range_connected_unitaries(n_qubits, reps, max_range, batch_size,
                                                 device=device, dtype=dtype, generator=generator,
                                                 samples_per_wiring=samples_per_wiring)
    return sampler


def check_sane(n_qubits=4, reps=8, t=2, n_samples=1500, samples_per_wiring=100):
    """Fast sanity check before the sweep: unitary, det=1, and roughly-Haar
    at generous depth with unrestricted range -- catches a construction bug
    as a loud, obvious failure instead of a confusing sweep table."""
    print(f"--- sanity: n_qubits={n_qubits}, reps={reps}, unrestricted range ---")
    max_range = n_qubits - 1
    U = sample_range_connected_unitaries(n_qubits, reps, max_range, batch_size=4,
                                          dtype=torch.complex128, samples_per_wiring=4)
    d = 2 ** n_qubits
    unit_err = (U.conj().transpose(-1, -2) @ U
                - torch.eye(d, dtype=torch.complex128, device=U.device)).abs().max().item()
    det_err = (torch.linalg.det(U).abs() - 1).abs().max().item()
    print(f"  unitarity error={unit_err:.2e}  |det|-1 error={det_err:.2e}")

    est = fp.estimate_once_from_sampler(_sampler(n_qubits, reps, max_range, samples_per_wiring),
                                         d, t, n_samples, dtype=torch.complex128)
    print(f"  F={est.frame_potential:.4f}  Haar={est.haar:.1f}  ratio={est.ratio:.4f}  "
          f"(expect close to 1.0 -- generous depth, no connectivity restriction)")
    print()


def sweep_gain(n_qubits=6, reps_list=(2, 4, 8), ranges=(1, 2, 5), t=2,
               n_samples=1500, samples_per_wiring=100):
    """The actual point of this module: F^(t)/Haar ratio for every
    (max_range, reps) combination, at matched gate count -- read down a
    column to see the gain from more depth at fixed range, read across a
    row to see the gain from more reach at fixed depth."""
    d = 2 ** n_qubits
    haar = math.factorial(t)
    print(f"--- connectivity-range sweep: n_qubits={n_qubits}, t={t} (Haar F^(t)={haar}) ---")
    header = "reps".ljust(6) + "".join(f"range<={r}".rjust(14) for r in ranges)
    print(header)
    for reps in reps_list:
        row = str(reps).ljust(6)
        for r in ranges:
            est = fp.estimate_once_from_sampler(_sampler(n_qubits, reps, r, samples_per_wiring),
                                                 d, t, n_samples, dtype=torch.complex64)
            row += f"{est.ratio:14.4f}"
        print(row)
    print("  (each cell: F^(t)/Haar ratio -- 1.0 is exact Haar; smaller reps needed")
    print("   to reach a given ratio at larger range is the 'gain' from more connectivity)")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--reps", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--ranges", type=int, nargs="+", default=None,
                    help="max connectivity ranges to compare; default spans "
                         "[1 (local), a middle value, n_qubits-1 (fully connected)]")
    p.add_argument("--t", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--samples-per-wiring", type=int, default=100)
    args = p.parse_args(argv)

    ranges = args.ranges
    if ranges is None:
        ranges = sorted({1, max(1, args.n_qubits // 2), args.n_qubits - 1})

    check_sane(n_qubits=min(args.n_qubits, 4), samples_per_wiring=args.samples_per_wiring)
    sweep_gain(n_qubits=args.n_qubits, reps_list=args.reps, ranges=ranges, t=args.t,
               n_samples=args.n_samples, samples_per_wiring=args.samples_per_wiring)


if __name__ == "__main__":
    main()
