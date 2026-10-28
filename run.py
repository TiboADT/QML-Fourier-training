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

Sweep every non-redundant generic_ansatz architecture (see
generic_ansatz.list_architectures()) instead of the hardcoded circuit_set
numbers -- pass --generic, and --circuits all to use every one of them
(the default when --generic is given with no --circuits):
    python run.py frame-potential --generic --n-qubits 6 --reps 1 2 3 --t 2
    python run.py train --generic --circuits all --n-qubits 6 --layers 3 --reps 1 --max-steps 600

Run many (circuit, reps, ...) combos concurrently across CPU cores instead
of one at a time -- --n-workers, roughly matched to the machine's core
count (each individual run is too small to use more than one core by
itself, see --n-workers' own --help, so this is the lever that actually
uses the rest of a many-core machine on a big sweep):
    python run.py frame-potential --generic --n-qubits 6 --max-params 600 --t 2 --n-workers 16
    python run.py train --generic --n-qubits 6 --layers 3 --max-params 600 --n-workers 16
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import torch
import torch.multiprocessing

import frame_potential as fp
import generic_ansatz
from circuits import n_trainable

torch.multiprocessing.set_sharing_strategy("file_system")


def _parallel_map(fn, arg_tuples, n_workers):
    """Apply fn(*args) for each args in arg_tuples, yielding (original_index,
    result) pairs as they complete -- not necessarily in submission order,
    since that's the point of running them concurrently (see
    concurrent.futures.as_completed). n_workers <= 1 runs sequentially in
    this process instead (no process-pool overhead, and exactly today's
    single-worker behavior/output ordering).

    Processes, not threads: Python's GIL means only one thread executes
    Python bytecode at a time, and this project's per-circuit work is
    dominated by Python-level overhead (building up a gate-by-gate tape,
    looping over wires) rather than large tensor math, so threads wouldn't
    actually run concurrently here. Separate OS processes are what
    sidesteps the GIL for CPU-bound work like this.

    `fn` must be a module-level function (not a local closure or lambda --
    those aren't reliably picklable for sending to a worker process), and
    every item in arg_tuples must itself be picklable (plain tensors/numbers
    are fine).

    One task raising is not fatal to the rest of the sweep: its exception is
    yielded in place of a result (callers check `isinstance(result,
    Exception)`) so the caller can report it and move on. Before this, one
    bad task (a transient CUDA OOM, say) propagated out of future.result()
    uncaught, which aborted the whole loop immediately -- discovered the
    hard way on a long --max-params sweep where that meant losing every
    other task's output too, including ones that had already finished:
    record_training (which creates the CSV in the first place) never got a
    chance to run for *anything*.
    """
    if n_workers <= 1:
        for i, args_tuple in enumerate(arg_tuples):
            try:
                yield i, fn(*args_tuple)
            except Exception as e:
                yield i, e
        return
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fn, *args_tuple): i for i, args_tuple in enumerate(arg_tuples)}
        for future in as_completed(futures):
            try:
                yield futures[future], future.result()
            except Exception as e:
                yield futures[future], e


def parse_circuits(values, generic: bool = False):
    """Accept individual numbers and/or ranges like '1-19' in the same list
    (legacy circuit_set numbers).

    generic=True: values are generic_ansatz codes instead -- ranges aren't
    meaningful for those (codes aren't sequential), so each value is a
    single code, except the literal 'all', which expands to
    generic_ansatz.list_architectures() (every distinct, non-redundant
    architecture)."""
    if generic:
        if list(values) == ["all"]:
            return generic_ansatz.list_architectures()
        return [int(v) for v in values]
    circuits = []
    for v in values:
        if "-" in v:
            lo, hi = v.split("-")
            circuits.extend(range(int(lo), int(hi) + 1))
        else:
            circuits.append(int(v))
    return circuits


# ── train ───────────────────────────────────────────────────────────────

def _train_one(x, y, circuit_num, n_qubits, layers, reps, max_steps, batch_size, notes, generic):
    """One (circuit, n_qubits, layers, reps) training run -- module-level
    and self-contained (imports its own dependencies) so it can be sent to
    a worker process by _parallel_map. Pure compute via run_training, no
    file I/O (that happens back in the main process via record_training --
    see run_training's own docstring for why).

    Blinds this process to CUDA first (see cmd_train's identical line for
    why): training never asks for a GPU anywhere in its own code (no
    --device flag even exists for `train`, unlike frame-potential), but on
    a machine that has one, recent PyTorch optimizer internals probe
    torch.accelerator.current_stream() on every .step() regardless -- which
    tries to retain a CUDA context purely for that probe. With --n-workers
    spawning many concurrent processes each doing this, that's enough
    *context* overhead alone (no real tensors on it) to OOM the GPU, for a
    device training was never going to use. Set again here (not just in
    cmd_train) in case a future multiprocessing start method stops
    inheriting the parent's already-modified environment the way today's
    default ('fork' on Linux) does.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import time as _time
    from experiment_tracker import run_training
    t0 = _time.time()
    result = run_training(
        x, y, circuit_num=circuit_num, n_qubits=n_qubits, layers=layers, anzats_reps=reps,
        max_steps=max_steps, batch_size=batch_size, notes=notes, generic=generic,
    )
    # record_training (the only thing cmd_train does with the returned
    # result) never reads final_weights -- dropping it here
    result.final_weights = None
    return result, _time.time() - t0


def add_train_parser(sub):
    p = sub.add_parser("train", help="train circuit_set (or generic_ansatz) architectures on Fourier-series targets")
    p.add_argument("--circuits", type=str, nargs="+", default=None,
                    help="circuit numbers, e.g. '7' or '1-19' (mixable, space-separated). "
                         "With --generic, each value is a generic_ansatz code instead "
                         "(ranges aren't meaningful there), or 'all' for every "
                         "non-redundant architecture from generic_ansatz.list_architectures(). "
                         "Default: '1-19' normally, 'all' with --generic.")
    p.add_argument("--generic", action="store_true",
                    help="interpret --circuits as generic_ansatz codes instead of "
                         "circuit_set numbers -- see generic_ansatz.py's module docstring")
    p.add_argument("--n-qubits", type=int, nargs="+", default=[6])
    p.add_argument("--layers", type=int, nargs="+", default=[3],
                    help="number of data-reuploading layers -- stays fixed per sweep "
                         "point even with --max-params (see --max-params); it's the "
                         "reps budget-sweep that mirrors frame-potential's, not layers")
    p.add_argument("--reps", type=int, nargs="+", default=None,
                    help="ansatz repetitions per layer (anzats_reps) to sweep "
                         "(default: 1 2 3). Mutually exclusive with --max-params.")
    p.add_argument("--max-params", type=int, default=None,
                    help="instead of a fixed --reps list, sweep reps=1,2,3,... "
                         "*per circuit* (at each fixed --layers value) for as long "
                         "as layers * circuits.n_trainable(num, n_qubits, reps) "
                         "stays within this budget -- same idea as "
                         "frame-potential's --max-params, so a circuit's training "
                         "cost and its F^(t) can be compared at matched parameter "
                         "counts instead of matched reps. Mutually exclusive with --reps.")
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
    p.add_argument("--n-workers", type=int, default=1,
                    help="train this many (circuit, n_qubits, layers, reps) combos "
                         "concurrently, in separate OS processes (default: 1, i.e. "
                         "today's sequential behavior). Good value: close to the "
                         "machine's core count for a CPU run -- each run's own "
                         "compute is too small to use more than one core by itself "
                         "(see run.py's module docstring), so this is the lever that "
                         "actually uses the rest of the machine on a big sweep. All "
                         "file writing still happens in the main process (one writer, "
                         "no concurrent-CSV-append risk), only training runs in the "
                         "workers.")
    return p


def cmd_train(args):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from pennylane import numpy as pnp
    from experiment_tracker import record_training
    from functions import function_to_learn

    if args.seed is not None:
        pnp.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.reps is not None and args.max_params is not None:
        raise SystemExit("--reps and --max-params are mutually exclusive -- give "
                          "a fixed rep list or a parameter budget, not both.")

    circuits_arg = args.circuits if args.circuits is not None else (["all"] if args.generic else ["1-19"])
    circuits = parse_circuits(circuits_arg, generic=args.generic)

    if args.max_params is not None:
        # layers stays fixed per sweep point (the outer loop below); only
        # reps is budget-driven, same as frame-potential's --max-params --
        # see reps_for_param_budget's docstring for why layers is the
        # multiplier rather than another budget-swept dimension.
        # see cmd_frame_potential's comment on the identical loop there: this
        # calls n_trainable (a real PennyLane forward+backward pass) once per
        # reps tried, per circuit -- printing only on a skip left a long,
        # silent (and, before a fix elsewhere, occasionally infinite) wait
        # that was indistinguishable from a hang.
        combos = []
        for n_qubits in args.n_qubits:
            for layers in args.layers:
                for i, num in enumerate(circuits, start=1):
                    t_budget0 = time.time()
                    reps_list = reps_for_param_budget(num, n_qubits, args.max_params,
                                                        multiplier=layers, generic=args.generic)
                    print(f"  [{i}/{len(circuits)}] circuit {num} (n_qubits={n_qubits}, layers={layers}): "
                          f"reps={reps_list} ({time.time() - t_budget0:.1f}s)")
                    if not reps_list:
                        print(f"    skipped -- even reps=1 already has "
                              f"{layers * n_trainable(num, n_qubits, 1, generic=args.generic)} "
                              f"trainable parameters, over --max-params {args.max_params}")
                        continue
                    combos.extend((n_qubits, layers, reps, num) for reps in reps_list)
        print(f"degrees={args.degrees} n_functions={args.n_functions} circuits={circuits} "
              f"n_qubits={args.n_qubits} layers={args.layers} max_params={args.max_params} "
              f"-> {len(combos)} (n_qubits, layers, reps, circuit) combos")
    else:
        reps_list = args.reps if args.reps is not None else [1, 2, 3]
        combos = list(product(args.n_qubits, args.layers, reps_list, circuits))
        print(f"degrees={args.degrees} n_functions={args.n_functions} circuits={circuits} "
              f"n_qubits={args.n_qubits} layers={args.layers} reps={reps_list} -> "
              f"{len(combos)} (n_qubits, layers, reps, circuit) combos")

    n_runs = len(args.degrees) * args.n_functions * len(combos)

    # Build the full flat work list up front -- (x, y) per (degree, fn_i) is
    # shared across every combo trained on it (so all architectures for one
    # target draw are compared fairly), generated once here rather than
    # per-combo. `meta` carries the (degree, fn_i, n_qubits, layers, reps,
    # num) label for each item, same indexing as `arg_tuples`, since the
    # worker function itself doesn't need degree/fn_i (only used for the
    # printed label) -- see _parallel_map/_train_one.
    meta = []
    arg_tuples = []
    for degree in args.degrees:
        for fn_i in range(args.n_functions):
            target = function_to_learn(degree=degree)
            x = torch.linspace(-torch.pi, torch.pi, steps=args.n_samples, requires_grad=False)
            with torch.no_grad():
                y = target(x)
            for n_qubits, layers, reps, num in combos:
                meta.append((degree, fn_i, n_qubits, layers, reps, num))
                arg_tuples.append((x, y, num, n_qubits, layers, reps,
                                    args.max_steps, args.batch_size, args.notes, args.generic))

    t0 = time.time()
    n_failed = 0
    for i, outcome in _parallel_map(_train_one, arg_tuples, args.n_workers):
        degree, fn_i, n_qubits, layers, reps, num = meta[i]
        if isinstance(outcome, Exception):
            n_failed += 1
            print(f"[{i + 1}/{n_runs}] FAILED degree={degree} fn={fn_i} circuit={num} "
                  f"n_qubits={n_qubits} layers={layers} reps={reps}: {outcome!r}")
            continue
        result, elapsed = outcome
        record_training(result, path=args.out)
        if not args.quiet:
            print(f"[{i + 1}/{n_runs}] ({elapsed:.1f}s) degree={degree} fn={fn_i} "
                  f"circuit={num} n_qubits={n_qubits} layers={layers} reps={reps} "
                  f"final_cost={result.cst[-1].item():.6f}")

    failed_note = f", {n_failed} failed (see FAILED lines above)" if n_failed else ""
    print(f"\nDone: {n_runs - n_failed}/{n_runs} runs recorded in {time.time() - t0:.1f}s"
          f"{failed_note} -> {args.out}")


# ── frame-potential ─────────────────────────────────────────────────────

def _fp_one(num, n_qubits, reps, t, converge, n_samples, rel_tol, max_batches,
            device, dtype, seed, generic):
    """One (circuit, reps, t) frame-potential estimate -- module-level and
    self-contained so it can be sent to a worker process by _parallel_map
    (see _train_one's docstring for why). Already pure compute with no file
    I/O of its own -- fp.estimate_once/estimate_until_converged never
    touched disk, only the fp.save_estimate call in cmd_frame_potential
    does, which stays in the main process.

    GPU note: if --device cuda, every worker process opens its own CUDA
    context on the *same* GPU -- several of them competing for one GPU's
    memory/compute usually won't beat a single process using a larger
    --n-samples batch (GPU throughput comes from batching within a process,
    not from running many small processes). --n-workers is for CPU runs.
    """
    import time as _time
    import torch as _torch
    import frame_potential as _fp
    t0 = _time.time()
    generator = _torch.Generator().manual_seed(seed) if seed is not None else None
    if converge:
        est = _fp.estimate_until_converged(
            num, n_qubits, reps, t, generic=generic,
            n_samples=n_samples, rel_tol=rel_tol, max_batches=max_batches,
            device=device, dtype=dtype, generator=generator, verbose=False,
        )
    else:
        n = n_samples or (2 ** n_qubits * t)
        est = _fp.estimate_once(num, n_qubits, reps, t, n, generic=generic,
                                 device=device, dtype=dtype, generator=generator)
    return est, _time.time() - t0


def add_frame_potential_parser(sub):
    p = sub.add_parser("frame-potential", help="estimate F^(t) for circuit_set (or generic_ansatz) architectures")
    p.add_argument("--circuits", type=str, nargs="+", default=None,
                    help="circuit numbers, e.g. '7' or '1-19' (mixable, space-separated). "
                         "With --generic, each value is a generic_ansatz code instead "
                         "(ranges aren't meaningful there), or 'all' for every "
                         "non-redundant architecture from generic_ansatz.list_architectures(). "
                         "Default: '1-19' normally, 'all' with --generic.")
    p.add_argument("--generic", action="store_true",
                    help="interpret --circuits as generic_ansatz codes instead of "
                         "circuit_set numbers -- see generic_ansatz.py's module docstring")
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
    p.add_argument("--n-workers", type=int, default=1,
                    help="estimate this many (circuit, reps, t) combos concurrently, "
                         "in separate OS processes (default: 1, i.e. today's sequential "
                         "behavior) -- see train's --n-workers help for why processes, "
                         "not threads. Intended for --device cpu: for cuda, several "
                         "worker processes share one GPU and usually lose to a single "
                         "process with a bigger --n-samples batch instead (see _fp_one's "
                         "docstring). With --converge and --n-workers > 1, the "
                         "already-exhausted-at-a-smaller-reps skip (`stuck`) is not "
                         "applied across workers -- each combo runs independently, so a "
                         "few extra already-doomed convergence attempts (bounded by "
                         "--max-batches each) can happen instead of being skipped.")
    return p


def reps_for_param_budget(num, n_qubits, max_params, multiplier=1, generic: bool = False):
    """All reps = 1, 2, 3, ... whose multiplier * circuits.n_trainable(num,
    n_qubits, reps) stays at or under max_params -- not the raw allocated
    weight-tensor size (several circuits don't read all of it, see
    n_trainable's own docstring), so circuits compared at the same
    max_params are matched on real degrees of freedom. Returns the full
    list (not just the largest reps) so a sweep shows how F^(t) (or the
    training cost) evolves with depth up to the budget, not just the
    endpoint. Empty if even reps=1 already exceeds max_params.

    `multiplier` is 1 for frame-potential (no layers concept there -- see
    frame_potential.sample_unitaries, it's just circuit_set(num, reps)) and
    `layers` for training (build_model stacks `layers` independent copies
    of the same reps-parametrized block, each with its own weights -- see
    experiment_tracker.py's n_params = layers * n_trainable(...) -- so the
    *total* trainable-parameter count a training run actually has is that
    product, not n_trainable alone).

    generic=True: `num` is a generic_ansatz code -- see circuits.circuit_set.
    Some generic_ansatz codes (topo_category=0 -- no entangling topology at
    all, e.g. 20000) deliberately collapse to the same fixed-size single
    rotation layer for *every* reps (see generic_ansatz's module docstring),
    so n_trainable never grows and never exceeds max_params -- without the
    plateau check below this would loop forever instead of stopping once
    increasing reps stops changing anything."""
    reps_list = []
    reps = 1
    prev_n = None
    while True:
        n = multiplier * n_trainable(num, n_qubits, reps, generic=generic)
        if n > max_params or n == prev_n:
            break
        reps_list.append(reps)
        prev_n = n
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

    circuits_arg = args.circuits if args.circuits is not None else (["all"] if args.generic else ["1-19"])
    circuits = parse_circuits(circuits_arg, generic=args.generic)
    device = torch.device(args.device) if args.device else fp.get_device()
    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    if args.max_params is not None:
        # Building this sweep means calling n_trainable (a real forward+backward
        # pass through a PennyLane QNode) once per reps value tried, per
        # circuit -- for a large --circuits list (e.g. --generic with no
        # --circuits, which sweeps every architecture) that can take tens of
        # seconds with nothing else printed yet, which looks identical to a
        # hang. One line per circuit here, as it's computed, instead of only
        # ever printing on a skip, is the fix for that -- not just reassurance,
        # it also makes a genuinely slow or runaway architecture's reps count
        # (or render time) visible immediately instead of after the whole
        # list finishes.
        combos = []
        for i, num in enumerate(circuits, start=1):
            t_budget0 = time.time()
            reps_list = reps_for_param_budget(num, args.n_qubits, args.max_params, generic=args.generic)
            print(f"  [{i}/{len(circuits)}] circuit {num}: reps={reps_list} "
                  f"({time.time() - t_budget0:.1f}s)")
            if not reps_list:
                print(f"    skipped -- even reps=1 already has "
                      f"{n_trainable(num, args.n_qubits, 1, generic=args.generic)} trainable parameters, "
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

    n_runs = len(combos)
    t0 = time.time()

    if args.n_workers <= 1:
        # (num, t) pairs where a smaller reps already exhausted --max-batches
        # without satisfying estimate_until_converged's own stopping rule.
        # combos is built with reps non-decreasing for any fixed (num, t) (both
        # branches above sweep reps in increasing order per circuit), so once a
        # pair lands here every later occurrence is a strictly larger reps --
        # safe to skip: more reps can only move F^(t) closer to Haar, shrinking
        # delta further, which only makes the (already unmet) rel_tol*delta
        # target harder to hit, never easier. Re-running would just burn
        # another full --max-batches for no new information. Only tracked
        # sequentially -- see the --n-workers > 1 branch below for why.
        stuck = set()
        for i, (num, reps, t) in enumerate(combos, start=1):
            if args.converge and (num, t) in stuck:
                print(f"[{i}/{n_runs}] skipped: circuit {num} (t={t}) already exhausted "
                      f"--max-batches={args.max_batches} without converging at a smaller reps -- "
                      "more reps won't change that, only raising --max-batches would.")
                continue
            run_t0 = time.time()
            if args.converge:
                est = fp.estimate_until_converged(
                    num, args.n_qubits, reps, t, generic=args.generic,
                    n_samples=args.n_samples, rel_tol=args.rel_tol, max_batches=args.max_batches,
                    device=device, dtype=dtype, generator=generator, verbose=not args.quiet,
                )
                if not _converged(est, args.rel_tol):
                    stuck.add((num, t))
            else:
                n_samples = args.n_samples or (2 ** args.n_qubits * t)
                est = fp.estimate_once(
                    num, args.n_qubits, reps, t, n_samples, generic=args.generic,
                    device=device, dtype=dtype, generator=generator,
                )
            fp.save_estimate(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps, generic=args.generic,
                              device=device, dtype=dtype, seed=args.seed, notes=args.notes, path=args.out)
            if not args.quiet:
                print(f"[{i}/{n_runs}] ({time.time() - run_t0:.1f}s) \n" +
                      fp.report(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps, generic=args.generic))
    else:
        # No `stuck` skip across workers here (see --n-workers's own help):
        # each combo is dispatched independently, with no visibility into
        # sibling workers' outcomes. Every worker seeds its own generator
        # from the same --seed (can't share one stateful Generator object
        # across processes), so this branch's draws don't reproduce the
        # n_workers<=1 branch's even at the same --seed -- same caveat
        # cmd_train's --seed help already makes for that command.
        arg_tuples = [
            (num, args.n_qubits, reps, t, args.converge, args.n_samples, args.rel_tol,
             args.max_batches, device, dtype, args.seed, args.generic)
            for (num, reps, t) in combos
        ]
        n_failed = 0
        for i, outcome in _parallel_map(_fp_one, arg_tuples, args.n_workers):
            num, reps, t = combos[i]
            if isinstance(outcome, Exception):
                n_failed += 1
                print(f"[{i + 1}/{n_runs}] FAILED circuit {num} reps={reps} t={t}: {outcome!r}")
                continue
            est, elapsed = outcome
            fp.save_estimate(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps, generic=args.generic,
                              device=device, dtype=dtype, seed=args.seed, notes=args.notes, path=args.out)
            if not args.quiet:
                print(f"[{i + 1}/{n_runs}] ({elapsed:.1f}s) \n" +
                      fp.report(est, circuit_num=num, n_qubits=args.n_qubits, reps=reps, generic=args.generic))
        failed_note = f", {n_failed} failed (see FAILED lines above)" if n_failed else ""
        print(f"\nDone: {n_runs - n_failed}/{n_runs} runs recorded in {time.time() - t0:.1f}s"
              f"{failed_note} -> {args.out}")
        return

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
