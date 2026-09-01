"""
Single CLI entry point for this repo: training and frame-potential
estimation, as two subcommands sharing one argument-parsing/dispatch setup.

Examples
--------
Train a few architectures on one target Fourier function:
    python run.py train --circuits 1 7 11 --n-qubits 6 --layers 3 --reps 1 --max-steps 600

Full training sweep matching test.py's original defaults:
    python run.py train --circuits 1-19 30 31 32 --n-qubits 6 --layers 3 --reps 1 2 3 \\
        --degrees 10 --n-functions 5 --max-steps 600

Quick frame-potential check:
    python run.py frame-potential --circuits 7 --n-qubits 4 --reps 1 --t 2 --device cpu

Full frame-potential sweep, converged:
    python run.py frame-potential --circuits 1-19 --n-qubits 6 --reps 1 2 3 --t 2 --converge --seed 0
"""

import argparse
import time
from itertools import product

import torch

import frame_potential as fp


def parse_circuits(values):
    """Accept individual numbers and/or ranges like '1-19' in the same list."""
    circuits = []
    for v in values:
        if "-" in v:
            lo, hi = v.split("-")
            circuits.extend(range(int(lo), int(hi) + 1))
        else:
            circuits.append(int(v))
    return circuits


# ── train ───────────────────────────────────────────────────────────────

def add_train_parser(sub):
    p = sub.add_parser("train", help="train circuit_set architectures on Fourier-series targets")
    p.add_argument("--circuits", type=str, nargs="+", default=["1-19"],
                    help="circuit numbers, e.g. '7' or '1-19' (mixable, space-separated)")
    p.add_argument("--n-qubits", type=int, nargs="+", default=[6])
    p.add_argument("--layers", type=int, nargs="+", default=[3],
                    help="number of data-reuploading layers")
    p.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3],
                    help="ansatz repetitions per layer (anzats_reps)")
    p.add_argument("--degrees", type=int, nargs="+", default=[10],
                    help="degree(s) of the target Fourier series")
    p.add_argument("--n-functions", type=int, default=5,
                    help="how many random target functions to draw per degree")
    p.add_argument("--n-samples", type=int, default=800,
                    help="x points sampled on [-pi, pi] per target function")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=None,
                    help="best-effort reproducibility: seeds target-function "
                         "generation and training; does not make the whole "
                         "sweep bit-for-bit deterministic")
    p.add_argument("--out", default="results/", help="directory for experiments.csv / costs.csv")
    p.add_argument("--notes", default="")
    p.add_argument("--quiet", action="store_true")
    return p


def cmd_train(args):
    from pennylane import numpy as pnp
    from experiment_tracker import train_and_record
    from functions import function_to_learn

    if args.seed is not None:
        pnp.random.seed(args.seed)
        torch.manual_seed(args.seed)

    circuits = parse_circuits(args.circuits)
    combos = list(product(args.n_qubits, args.layers, args.reps, circuits))
    n_runs = len(args.degrees) * args.n_functions * len(combos)
    print(f"degrees={args.degrees} n_functions={args.n_functions} circuits={circuits} "
          f"n_qubits={args.n_qubits} layers={args.layers} reps={args.reps} -> {n_runs} runs")

    t0 = time.time()
    run_i = 0
    for degree in args.degrees:
        for fn_i in range(args.n_functions):
            target = function_to_learn(degree=degree)
            x = torch.linspace(-torch.pi, torch.pi, steps=args.n_samples, requires_grad=False)
            with torch.no_grad():
                y = target(x)

            for n_qubits, layers, reps, num in combos:
                run_i += 1
                run_t0 = time.time()
                exp_id, weights, cst = train_and_record(
                    x, y, circuit_num=num, n_qubits=n_qubits, layers=layers, anzats_reps=reps,
                    max_steps=args.max_steps, batch_size=args.batch_size,
                    notes=args.notes, path=args.out,
                )
                if not args.quiet:
                    print(f"[{run_i}/{n_runs}] ({time.time() - run_t0:.1f}s) degree={degree} fn={fn_i} "
                          f"circuit={num} n_qubits={n_qubits} layers={layers} reps={reps} "
                          f"final_cost={cst[-1].item():.6f}")

    print(f"\nDone: {n_runs} runs in {time.time() - t0:.1f}s -> {args.out}")


# ── frame-potential ─────────────────────────────────────────────────────

def add_frame_potential_parser(sub):
    p = sub.add_parser("frame-potential", help="estimate F^(t) for circuit_set architectures")
    p.add_argument("--circuits", type=str, nargs="+", default=["1-19"],
                    help="circuit numbers, e.g. '7' or '1-19' (mixable, space-separated)")
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--t", type=int, nargs="+", default=[2])
    p.add_argument("--n-samples", type=int, default=None,
                    help="samples per batch (default: 2**n_qubits * t)")
    p.add_argument("--converge", action="store_true",
                    help="keep pooling batches until the 95%% CI is tight (see --rel-tol); "
                         "default is a single batch of --n-samples")
    p.add_argument("--rel-tol", type=float, default=0.4)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="default: cuda if available, else cpu")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=fp.FRAME_POTENTIAL_CSV)
    p.add_argument("--notes", default="")
    p.add_argument("--quiet", action="store_true", help="suppress the per-run report")
    return p


def cmd_frame_potential(args):
    circuits = parse_circuits(args.circuits)
    device = torch.device(args.device) if args.device else fp.get_device()
    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    print(f"device={device} dtype={dtype} circuits={circuits} n_qubits={args.n_qubits} "
          f"reps={args.reps} t={args.t} converge={args.converge}")

    n_runs = len(circuits) * len(args.reps) * len(args.t)
    t0 = time.time()
    for i, (num, reps, t) in enumerate(product(circuits, args.reps, args.t), start=1):
        run_t0 = time.time()
        if args.converge:
            est = fp.estimate_until_converged(
                num, args.n_qubits, reps, t,
                n_samples=args.n_samples, rel_tol=args.rel_tol, max_batches=args.max_batches,
                device=device, dtype=dtype, generator=generator, verbose=not args.quiet,
            )
        else:
            n_samples = args.n_samples or (2 ** args.n_qubits * t)
            est = fp.estimate_once(
                num, args.n_qubits, reps, t, n_samples,
                device=device, dtype=dtype, generator=generator,
            )
        fp.save_estimate(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps,
                          device=device, dtype=dtype, seed=args.seed, notes=args.notes, path=args.out)
        if not args.quiet:
            print(f"[{i}/{n_runs}] ({time.time() - run_t0:.1f}s) \n" +
                  fp.report(est, circuit_num=num, n_qubits=args.n_qubits))

    print(f"\nDone: {n_runs} runs in {time.time() - t0:.1f}s -> {args.out}")


# ── dispatch ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    add_train_parser(sub)
    add_frame_potential_parser(sub)
    args = p.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "frame-potential":
        cmd_frame_potential(args)


if __name__ == "__main__":
    main()
