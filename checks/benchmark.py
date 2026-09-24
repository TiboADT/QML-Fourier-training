"""
Benchmarks for Circuits_training. Run from the repo root:

    python check.py benchmark                # everything available
    python check.py benchmark --only fp      # frame-potential only
    python check.py benchmark --only train   # training only
    python check.py benchmark --only stress  # frame potential at higher n_qubits + convergence time

`stress` is deliberately separate from `fp`: `fp` is a quick regression
check (small, fixed sizes, every device, run as part of the default
everything-sweep); `stress` is for deciding how far a *specific* machine
can actually push n_qubits, so it isn't part of the default sweep and
defaults to the single best available device rather than looping cpu+cuda
(cpu at n_qubits=12+ would just time out for no information gained -- pass
--device cpu explicitly for a comparison point).

    python check.py benchmark --only stress --device cuda
    python check.py benchmark --only stress --n-qubits 8 10 12 14
    python check.py benchmark --only stress --circuits 1 18 34 --reps 2
"""
 
import argparse
import time
 
import torch
 
 
def timeit(fn, repeats=3, warmup=1):
    """Median wall-clock seconds, with CUDA synchronisation if relevant."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]
 
 
def header(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
 
 
# ── 1. frame potential: einsum vs GEMM ─────────────────────────────────────
 
def bench_frame_potential(device):
    header(f"1. Pairwise traces — einsum vs GEMM   [{device}]")
    print(f"{'n_qubits':>9} {'N':>6} {'einsum':>10} {'GEMM':>10} {'speedup':>9} "
          f"{'einsum peak':>12} {'GEMM peak':>10}")
 
    for n_qubits, N in [(4, 200), (6, 128), (6, 256), (8, 128)]:
        d = 2 ** n_qubits
        UA = torch.randn(N, d, d, dtype=torch.complex64, device=device)
        UB = torch.randn(N, d, d, dtype=torch.complex64, device=device)
 
        def with_einsum():
            A, B = UA.unsqueeze(1), UB.unsqueeze(0)
            return torch.einsum("bipq,bjpq->bij", A.conj(), B).squeeze(1)
 
        def with_gemm():
            return UA.reshape(N, -1).conj() @ UB.reshape(N, -1).T
 
        try:
            t_ein = timeit(with_einsum)
        except RuntimeError as e:                       # OOM is itself the result
            print(f"{n_qubits:>9} {N:>6} {'OOM':>10} "
                  f"{timeit(with_gemm)*1000:>9.2f}ms {'-':>9}   ({str(e)[:30]})")
            continue
        t_gem = timeit(with_gemm)
 
        err = (with_einsum() - with_gemm()).abs().max().item()
        peak_ein = N * N * d * d * 8 / 1e9
        peak_gem = (2 * N * d * d * 8 + N * N * 8) / 1e9
        print(f"{n_qubits:>9} {N:>6} {t_ein*1000:>9.2f}ms {t_gem*1000:>9.2f}ms "
              f"{t_ein/t_gem:>8.1f}x {peak_ein:>11.2f}GB {peak_gem:>9.3f}GB"
              f"   (max|diff| {err:.1e})")
 
    print("\nLargest N that fits an 8 GB budget at 6 qubits:")
    import math
    d, bpe, usable = 64, 8, 8e9 * 0.5
    a, b = bpe, 2 * d * d * bpe
    print(f"  current  (N,N,d,d)      : N = {int(math.sqrt(usable / (d*d*bpe)))}")
    print(f"  GEMM     (N,d^2) + (N,N): N = {int((-b + math.sqrt(b*b + 4*a*usable)) / (2*a))}")
 
 
def bench_sample_unitaries(device):
    header(f"2. sample_unitaries throughput   [{device}]")
    try:
        import frame_potential as fp
    except ImportError as e:
        print(f"  skipped ({e}) — run from the repo root")
        return
    print(f"{'circuit':>8} {'reps':>5} {'N':>6} {'time':>10} {'per unitary':>13}")
    for num, reps, N in [(18, 1, 128), (18, 3, 128), (5, 3, 128)]:
        try:
            t = timeit(lambda: fp.sample_unitaries(num, 6, reps, N, device=device))
            print(f"{num:>8} {reps:>5} {N:>6} {t*1000:>9.1f}ms {t/N*1e6:>11.1f}us")
        except Exception as e:
            print(f"{num:>8} {reps:>5} {N:>6}   failed: {str(e)[:40]}")
 
 
# ── 2b. frame potential: how far does this machine actually reach ─────────

def bench_stress(device, n_qubits_list=None, circuits=None, reps=1, t=2):
    """End-to-end cost at scale: sample_unitaries throughput, one fixed-size
    estimate_once batch, and a full estimate_until_converged run, all at
    the SAME n_qubits -- so you can see where the unitary-construction cost
    (O(batch*d^2) per gate, see frame_potential.apply_embedded_gate) starts
    dominating versus where it's the pairwise-trace step (O(N^2*d^2), see
    bench_frame_potential above) or just the number of batches the
    convergence loop needs. `recommended_batch_size` caps every batch to
    what should fit in memory; a size that still OOMs is itself useful
    information (reported, not treated as an error) about where the actual
    ceiling on this machine is versus what the heuristic predicts.
    """
    header(f"2b. Frame-potential stress test — scaling with n_qubits   [{device}]")
    import frame_potential as fp

    if n_qubits_list is None:
        n_qubits_list = [8, 10, 12]
    if circuits is None:
        circuits = [1, 18]  # cheap (no entangling gates) vs. expensive (all-to-all CZ)

    print(f"{'circuit':>7} {'n_qubits':>9} {'d':>7} {'batch_cap':>10} "
          f"{'sample_U':>10} {'estimate_once':>14} {'converged':>11} "
          f"{'n_pairs':>12} {'fid_err':>9}")

    for num in circuits:
        for n_qubits in n_qubits_list:
            d = 2 ** n_qubits
            row = f"{num:>7} {n_qubits:>9} {d:>7}"
            try:
                max_batch = fp.recommended_batch_size(n_qubits, device)
            except Exception as e:
                print(f"{row}   recommended_batch_size failed: {str(e)[:40]}")
                continue
            row += f" {max_batch:>10}"

            try:
                sample_batch = max(2, min(32, max_batch))
                t_sample = timeit(
                    lambda: fp.sample_unitaries(num, n_qubits, reps, sample_batch, device=device),
                    repeats=1, warmup=1,
                )
                row += f" {t_sample*1000:>9.1f}ms"
            except RuntimeError as e:
                print(f"{row}   sample_unitaries OOM/failed: {str(e)[:40]}")
                continue

            try:
                n_samples_once = max(4, min(4 * d, max_batch))
                t0 = time.perf_counter()
                fp.estimate_once(num, n_qubits, reps, t, n_samples_once, device=device)
                t_once = time.perf_counter() - t0
                row += f" {t_once:>13.2f}s"
            except RuntimeError as e:
                print(f"{row}   estimate_once OOM/failed: {str(e)[:40]}")
                print(row)
                continue

            try:
                t0 = time.perf_counter()
                est = fp.estimate_until_converged(num, n_qubits, reps, t, device=device)
                t_conv = time.perf_counter() - t0
                row += f" {t_conv:>10.2f}s {est.n_pairs:>12,} {est.fidelity_error:>9.4f}"
            except RuntimeError as e:
                row += f"   estimate_until_converged OOM/failed: {str(e)[:40]}"

            print(row)
    print()


# ── 3. training ─────────────────────────────────────────────────────────────
 
def bench_square_loss():
    header("3. square_loss — Python loop vs vectorised (forward + backward)")
    for n in [100, 800]:
        pred = torch.rand(n, requires_grad=True)
        targ = torch.rand(n)
 
        def looped():
            loss = 0
            for a, b in zip(targ, pred):
                loss += (a - b) ** 2
            (0.5 * loss / len(targ)).backward()
            pred.grad = None
 
        def vector():
            (0.5 * ((targ - pred) ** 2).mean()).backward()
            pred.grad = None
 
        t1, t2 = timeit(looped, repeats=5), timeit(vector, repeats=5)
        print(f"  n={n:<5} looped {t1*1000:8.3f} ms   vectorised {t2*1000:8.3f} ms"
              f"   -> {t1/t2:6.1f}x")
 
 
def bench_training_step(devices):
    header("4. One training step, by device and batch size")
    try:
        from functions import build_model, square_loss
    except ImportError as e:
        print(f"  skipped ({e}) — run from the repo root")
        return
 
    for dev_name in devices:
        dev = torch.device(dev_name)
        try:
            model, weights = build_model(18, 6, layers=3, anzats_reps=3, measuring_qubit=5)
        except Exception as e:
            print(f"  {dev_name}: build_model failed: {str(e)[:60]}")
            continue
        weights = weights.detach().to(dev).requires_grad_(True)
 
        print(f"\n  device = {dev_name}")
        print(f"  {'batch':>7} {'forward':>11} {'fwd+bwd':>11} {'per step':>11} "
              f"{'proj. 600 steps':>16}")
        for batch in [50, 100, 400, 800]:
            x = torch.linspace(-torch.pi, torch.pi, batch, device=dev)
            y = torch.rand(batch, device=dev)
 
            def fwd():
                with torch.no_grad():
                    model(weights, x)
 
            def fwd_bwd():
                if weights.grad is not None:
                    weights.grad = None
                square_loss(y, model(weights, x)).backward()
 
            try: 
                tf, tb = timeit(fwd), timeit(fwd_bwd)
            except Exception as e:
                print(f"  {batch:>7}   failed: {str(e)[:50]}")
                continue
            print(f"  {batch:>7} {tf*1000:>10.1f}ms {tb*1000:>10.1f}ms "
                  f"{tb*1000:>10.1f}ms {tb*600:>15.1f}s")
 
 
def bench_devices():
    header("5. PennyLane device comparison (forward pass, 6 qubits)")
    try:
        import pennylane as qp
        from circuits import circuit_set, weight_tensor_shape
    except ImportError as e:
        print(f"  skipped ({e})")
        return
 
    n, reps, layers = 6, 3, 3
    shape = (layers,) + weight_tensor_shape(18, n, reps)
    w = 2 * torch.pi * torch.rand(shape)
    x = torch.linspace(-torch.pi, torch.pi, 800)
 
    for dev_name in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
        try:
            dev = qp.device(dev_name, wires=n)
        except Exception as e:
            print(f"  {dev_name:<18} unavailable ({str(e)[:45]})")
            continue
 
        @qp.qnode(dev, interface="torch")
        def circuit(weights, xs):
            f = circuit_set(num=18)
            f(weights[0], wires=list(range(n)))
            for l in range(layers - 1):
                for q in range(n):
                    qp.RX(xs, wires=q)
                f(weights[l + 1], wires=list(range(n)))
            return qp.expval(qp.PauliZ(n - 1))
 
        try:
            t = timeit(lambda: circuit(w, x), repeats=3)
            print(f"  {dev_name:<18} {t*1000:>9.1f} ms   ({t*600*1000:.0f} ms for 600 steps, fwd only)")
        except Exception as e:
            print(f"  {dev_name:<18} failed ({str(e)[:45]})")
 
 
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["fp", "train", "stress"], default=None)
    p.add_argument("--n-qubits", type=int, nargs="+", default=None,
                    help="stress only: n_qubits values to sweep (default: 8 10 12)")
    p.add_argument("--circuits", type=int, nargs="+", default=None,
                    help="stress only: circuit_set numbers to sweep (default: 1 18)")
    p.add_argument("--reps", type=int, default=1, help="stress only")
    p.add_argument("--t", type=int, default=2, help="stress only")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="stress only: default is the single best available device "
                         "(cuda if present), not cpu+cuda both -- pass this to force one")
    args = p.parse_args(argv)

    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print(f"threads: {torch.get_num_threads()}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    if args.only in (None, "fp"):
        for d in devices:
            bench_frame_potential(torch.device(d))
        bench_sample_unitaries(torch.device(devices[-1]))

    if args.only == "stress":
        import frame_potential as fp
        device = torch.device(args.device) if args.device else fp.get_device()
        bench_stress(device, n_qubits_list=args.n_qubits, circuits=args.circuits,
                     reps=args.reps, t=args.t)

    if args.only in (None, "train"):
        bench_square_loss()
        bench_training_step(devices)
        bench_devices()

    print("\nDone.")
 
 
if __name__ == "__main__":
    main()