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

By default this prints a text diagram sized to your actual terminal width
(not a fixed 200 columns regardless of terminal size -- the terminal
line-wrapping *that* caused, mid-diagram and independent of PennyLane's own
pagination, was most of why this used to be unreadable), with GlobalPhase
gates dropped: a global scalar phase is invisible to every physical
measurement and to frame_potential itself, so it carries no information a
circuit diagram needs -- and for the KAK1-based circuits (33-36) there's
one per block, which was the other major source of clutter (see
circuits.py's core_kak for why it's there at all: fixing up a 3-CNOT
core's determinant).

Pass --mpl to additionally save a Qiskit-style boxed diagram (matplotlib,
one gate per box, no line-wrapping) to a PNG -- better than ASCII can ever
be for a genuinely wide/dense circuit:

    python check.py show 34 --n-qubits 6 --reps 2 --mpl
    python check.py show 34 --n-qubits 6 --reps 2 --mpl my_circuit.png

Not covered: the Clifford-group ensemble (two_designs/clifford_group.py).
It's a stim stabilizer tableau, not a circuit_set-style gate sequence --
stim.Tableau.to_circuit() is the right tool if you want to look inside one,
and doesn't need PennyLane at all.
"""

import argparse
import shutil

import pennylane as qp
import torch

from circuits import circuit_set, weight_tensor_shape
from two_designs.range_connectivity import draw_display_circuit

# Purely-cosmetic ops to drop from the *diagram* only (never from the
# circuit actually being measured/estimated elsewhere) -- see module
# docstring for why GlobalPhase qualifies.
_SKIP_FOR_DISPLAY = ("GlobalPhase",)


def _build_tape(queue_fn):
    """Trace queue_fn() (a zero-arg callable that queues PennyLane ops --
    e.g. circuit_set(num)(weights, wires=...) with weights/wires already
    bound) into a QuantumTape with _SKIP_FOR_DISPLAY ops dropped. Shared by
    the text and mpl drawing paths below so both show exactly the same
    filtered circuit, and shaped like frame_potential._trace_operations
    (same AnnotatedQueue trick), just for display instead of computation."""
    with qp.queuing.AnnotatedQueue() as q:
        queue_fn()
    ops = [op for op in qp.tape.QuantumScript.from_queue(q).operations
           if op.name not in _SKIP_FOR_DISPLAY]
    return qp.tape.QuantumTape(ops, [qp.state()])


def _terminal_width(margin=2):
    """Actual terminal width, not a fixed guess -- see module docstring for
    why a fixed one was the main readability bug. Falls back to 100 columns
    when not attached to a real terminal (e.g. piped output)."""
    return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns - margin)


def _show(tape, title: str, mpl_path):
    print(title)
    print(qp.drawer.tape_text(tape, max_length=_terminal_width(), decimals=2))
    if mpl_path:
        fig, _ = qp.drawer.tape_mpl(tape, decimals=2)
        fig.savefig(mpl_path, dpi=150, bbox_inches="tight")
        print(f"\n(mpl diagram saved to {mpl_path})")


def show_circuit_set(num: int, n_qubits: int, reps: int, mpl_path):
    shape = weight_tensor_shape(num, n_qubits, reps)
    weights = 2 * torch.pi * torch.rand(shape, dtype=torch.float64)
    tape = _build_tape(lambda: circuit_set(num=num)(weights, wires=list(range(n_qubits))))
    title = f"circuit_set({num}), n_qubits={n_qubits}, reps={reps}, weight shape={tuple(shape)}\n"
    _show(tape, title, mpl_path)


def show_connectivity(n_qubits: int, reps: int, max_range: int, seed, mpl_path):
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

    tape = _build_tape(circuit_fn)
    _show(tape, "gate diagram:", mpl_path)


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
    p.add_argument("--mpl", nargs="?", const=True, default=None, metavar="PATH",
                    help="also save a matplotlib (boxed, Qiskit-style) diagram -- "
                         "to PATH if given, else circuit_<target>.png")
    args = p.parse_args(argv)

    mpl_path = f"results/circuit_{args.target}.png" if args.mpl is True else args.mpl

    if args.target in ("connectivity", "c", "range-connectivity"):
        show_connectivity(args.n_qubits, args.reps, args.max_range, args.seed, mpl_path)
        return

    try:
        num = int(args.target)
    except ValueError:
        raise SystemExit(f"'{args.target}' is neither a circuit_set number nor 'connectivity'.")
    show_circuit_set(num, args.n_qubits, args.reps, mpl_path)


if __name__ == "__main__":
    main()
