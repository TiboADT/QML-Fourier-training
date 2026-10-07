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

Compare architectures at a matched parameter budget instead of matched reps
(each circuit sweeps its own reps=1,2,3,... until circuits.n_trainable
would exceed the budget):
    python run.py frame-potential --circuits 1 18 34 --n-qubits 6 --max-params 100 --t 2
"""

import argparse
import time
from itertools import product

import torch

import frame_potential as fp
from circuits import n_trainable


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
    p.add_argument("--reps", type=int, nargs="+", default=None,
                    help="ansatz repetitions to sweep (default: 1 2 3). Mutually "
                         "exclusive with --max-params.")
    p.add_argument("--max-params", type=int, default=None,
                    help="instead of a fixed --reps list, sweep reps=1,2,3,... "
                         "*per circuit* for as long as circuits.n_trainable(num, "
                         "n_qubits, reps) stays within this budget -- lets "
                         "architectures with very different params-per-rep be "
                         "compared at a matched parameter count instead of a "
                         "matched rep count. Mutually exclusive with --reps.")
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


def reps_for_param_budget(num, n_qubits, max_params):
    """All reps = 1, 2, 3, ... whose circuits.n_trainable(num, n_qubits, reps)
    stays at or under max_params -- not the raw allocated weight-tensor size
    (several circuits don't read all of it, see n_trainable's own docstring),
    so circuits compared at the same max_params are matched on real degrees
    of freedom. Returns the full list (not just the largest reps) so a sweep
    shows how F^(t) evolves with depth up to the budget, not just the
    endpoint. Empty if even reps=1 already exceeds max_params."""
    reps_list = []
    reps = 1
    while True:
        if n_trainable(num, n_qubits, reps) > max_params:
            break
        reps_list.append(reps)
        reps += 1
    return reps_list


def _converged(est, rel_tol, min_abs_error=1e-5):
    """Re-derive whether estimate_until_converged's own stopping criterion
    was actually satisfied -- it returns only the final (possibly
    max_batches-exhausted) Estimate, not a flag saying which happened.
    Same formula as checks/validate_local_random.py's
    check_convergence_pathology; min_abs_error matches
    estimate_until_converged's own default since run.py doesn't expose
    that one as a flag."""
    target = abs(rel_tol * est.delta)
    return est.fidelity_error <= target or est.fidelity_error <= min_abs_error


def cmd_frame_potential(args):
    if args.reps is not None and args.max_params is not None:
        raise SystemExit("--reps and --max-params are mutually exclusive -- give "
                          "a fixed rep count or a parameter budget, not both.")

    circuits = parse_circuits(args.circuits)
    device = torch.device(args.device) if args.device else fp.get_device()
    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    if args.max_params is not None:
        combos = []
        for num in circuits:
            reps_list = reps_for_param_budget(num, args.n_qubits, args.max_params)
            if not reps_list:
                print(f"  circuit {num}: skipped -- even reps=1 already has "
                      f"{n_trainable(num, args.n_qubits, 1)} trainable parameters, "
                      f"over --max-params {args.max_params}")
                continue
            combos.extend(product([num], reps_list, args.t))
        print(f"device={device} dtype={dtype} circuits={circuits} n_qubits={args.n_qubits} "
              f"max_params={args.max_params} t={args.t} converge={args.converge}")
    else:
        reps_list = args.reps if args.reps is not None else [1, 2, 3]
        combos = list(product(circuits, reps_list, args.t))
        print(f"device={device} dtype={dtype} circuits={circuits} n_qubits={args.n_qubits} "
              f"reps={reps_list} t={args.t} converge={args.converge}")

    # (num, t) pairs where a smaller reps already exhausted --max-batches
    # without satisfying estimate_until_converged's own stopping rule.
    # combos is built with reps non-decreasing for any fixed (num, t) (both
    # branches above sweep reps in increasing order per circuit), so once a
    # pair lands here every later occurrence is a strictly larger reps --
    # safe to skip: more reps can only move F^(t) closer to Haar, shrinking
    # delta further, which only makes the (already unmet) rel_tol*delta
    # target harder to hit, never easier. Re-running would just burn
    # another full --max-batches for no new information.
    stuck = set()

    n_runs = len(combos)
    t0 = time.time()
    for i, (num, reps, t) in enumerate(combos, start=1):
        if args.converge and (num, t) in stuck:
            print(f"[{i}/{n_runs}] skipped: circuit {num} (t={t}) already exhausted "
                  f"--max-batches={args.max_batches} without converging at a smaller reps -- "
                  "more reps won't change that, only raising --max-batches would.")
            continue

        run_t0 = time.time()
        if args.converge:
            est = fp.estimate_until_converged(
                num, args.n_qubits, reps, t,
                n_samples=args.n_samples, rel_tol=args.rel_tol, max_batches=args.max_batches,
                device=device, dtype=dtype, generator=generator, verbose=not args.quiet,
            )
            if not _converged(est, args.rel_tol):
                stuck.add((num, t))
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
                  fp.report(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps))

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
