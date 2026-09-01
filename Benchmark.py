"""
Benchmarks for Circuits_training. Run from the repo root:
 
    python benchmark.py                # everything available
    python benchmark.py --only fp      # frame-potential only
    python benchmark.py --only train   # training only
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
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["fp", "train"], default=None)
    args = p.parse_args()
 
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    print(f"threads: {torch.get_num_threads()}")
 
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
 
    if args.only in (None, "fp"):
        for d in devices:
            bench_frame_potential(torch.device(d))
        bench_sample_unitaries(torch.device(devices[-1]))
 
    if args.only in (None, "train"):
        bench_square_loss()
        bench_training_step(devices)
        bench_devices()
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()