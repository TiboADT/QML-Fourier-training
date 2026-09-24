"""Show the actual gate structure of a circuit -- so "what am I testing"
has a direct answer instead of requiring a read through the code.

Two targets:
  - Any circuit_set architecture from circuits.py
  - One concrete random realization of the range-connectivity ensemble
    (two_designs/range_connectivity.py), which otherwise has no single
    fixed circuit to look at -- its wiring is redrawn per sample by
    design (see that module's docstring).

    python check.py show 34 --n-qubits 6 --reps 2
    python check.py show connectivity --n-qubits 6 --reps 3 --max-range 2 --seed 0

Not covered: the Clifford-group ensemble (two_designs/clifford_group.py).
It's a stim stabilizer tableau, not a circuit_set-style gate sequence --
stim.Tableau.to_circuit() is the right tool if you want to look inside one,
and doesn't need PennyLane at all.
"""

import argparse

import pennylane as qp
import torch

from circuits import circuit_set, weight_tensor_shape
from two_designs.range_connectivity import draw_display_circuit


def show_circuit_set(num: int, n_qubits: int, reps: int):
    shape = weight_tensor_shape(num, n_qubits, reps)
    weights = 2 * torch.pi * torch.rand(shape, dtype=torch.float64)
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit(w):
        circuit_set(num=num)(w, wires=list(range(n_qubits)))
        return qp.state()

    print(f"circuit_set({num}), n_qubits={n_qubits}, reps={reps}, weight shape={tuple(shape)}\n")
    print(qp.draw(circuit, max_length=200)(weights))


def show_connectivity(n_qubits: int, reps: int, max_range: int, seed=None):
    circuit_fn, wiring = draw_display_circuit(n_qubits, reps, max_range, seed=seed)

    seed_note = f"seed={seed}" if seed is not None else "unseeded -- a fresh random instance each run"
    print(f"range_connectivity: n_qubits={n_qubits}, reps={reps}, max_range={max_range} ({seed_note})\n")
    print("wiring drawn (one random_matching per layer):")
    for i, layer_pairs in enumerate(wiring):
        touched = {q for pair in layer_pairs for q in pair}
        idle = sorted(set(range(n_qubits)) - touched)
        note = f"   idle this layer: {idle}" if idle else ""
        print(f"  layer {i}: {layer_pairs}{note}")
    print()

    dev = qp.device("default.qubit", wires=n_qubits)
    qnode = qp.qnode(dev)(circuit_fn)
    print(qp.draw(qnode, max_length=200)())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="a circuit_set number (e.g. 34), or 'connectivity' for "
                                   "one range_connectivity realization")
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-range", type=int, default=1,
                    help="connectivity ensemble only; ignored for circuit_set numbers")
    p.add_argument("--seed", type=int, default=None,
                    help="connectivity ensemble only, for a reproducible wiring+angle draw")
    args = p.parse_args(argv)

    if args.target in ("connectivity", "c", "range-connectivity"):
        show_connectivity(args.n_qubits, args.reps, args.max_range, seed=args.seed)
        return

    try:
        num = int(args.target)
    except ValueError:
        raise SystemExit(f"'{args.target}' is neither a circuit_set number nor 'connectivity'.")
    show_circuit_set(num, args.n_qubits, args.reps)


if __name__ == "__main__":
    main()
