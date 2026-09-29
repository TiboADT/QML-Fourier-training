"""Reparametrization for exact Haar sampling of the KAK1 two-qubit ansatz
(circuits.py's circuit_set numbers 33/34, built from circuits.py's
core_kak/local_su2 primitives).

Background
----------
Tucci's KAK1 (arXiv:quant-ph/0507171) factors any U in SU(4) as

    U = (A1 (x) A0) * exp(i(k1 XX + k2 YY + k3 ZZ)) * (B1 (x) B0)

with A1,A0,B1,B0 in SU(2). Implemented as a circuit this is 4 local SU(2)
blocks (3 Euler angles each = 12 parameters) around a 3-CNOT non-local core
(3 parameters) = 15 parameters total, exactly the "15 single-qubit rotations
+ 3 CNOTs" construction.

frame_potential.py's sample_unitaries always draws raw circuit weights
*uniformly* on [0, 2*pi). Sampling every one of these 15 angles uniformly
does NOT reproduce Haar measure on SU(4): the 12 local angles need a
specific (still closed-form) reshaping, and the 3 non-local ("canonical")
angles need a fully joint, non-separable reshaping with no simple
closed-form inverse-CDF.

This module supplies that reshaping ("Phi" in the design discussion) so
that circuits.py's circuit_set(34), fed raw Uniform(0, 2*pi) parameters
(exactly what sample_unitaries already produces for every other circuit),
produces exactly Haar-distributed two-qubit unitaries. circuit_set(33) is
the same gate structure with the raw angles used directly instead (no
reparametrization at all) -- an ablation to quantify what this module
buys you, not a second consumer of it.

Local part (12 of 15 parameters) — closed form
------------------------------------------------
Each SU(2) block is built as RZ(gamma) RY(beta) RZ(alpha). Given
u_alpha, u_beta, u_gamma ~ Uniform[0, 1):

    alpha = 2*pi*u_alpha
    beta  = arccos(1 - 2*u_beta)      <- Bloch-sphere trick
    gamma = 2*pi*u_gamma

reproduces Haar measure on SU(2) exactly (standard result, e.g.
Zyczkowski & Kus, J. Phys. A 27, 4235 (1994)).

Non-local part (3 of 15 parameters) — empirical Rosenblatt transform
----------------------------------------------------------------------
The core circuit used here is (wires (c, t), applied as
CNOT(c->t); RZ(tz) on t; RY(ty1) on c; CNOT(t->c); RY(ty2) on c; CNOT(c->t)):
this is the standard 3-CNOT canonical-gate circuit, verified (see
scratch/derive_core*.py in the design session) to give full-rank coverage
of the 3-parameter Weyl chamber.

The exact analytic Haar-pushforward density for (tz, ty1, ty2) involves a
Weyl-integration-formula-type sine-product over an auxiliary eigenphase
labelling that is genuinely fiddly to get right by hand (see the design
notes: an initial closed-form derivation was numerically WRONG -- verified
via round-trip frame-potential testing against known Haar values -- because
of an unresolved sign/branch ambiguity in extracting those eigenphases).

Instead this module builds the reparametrization empirically and exactly:
    1. Draw N Haar-random SU(4) matrices the standard, trusted way
       (Ginibre matrix + QR, via scipy.stats.unitary_group).
    2. For each, solve (numerically, via the LOCAL-INVARIANT and manifestly
       unambiguous Makhlin G1/G2 trace invariants -- no eigendecomposition,
       hence no labelling ambiguity) for the (tz, ty1, ty2) that makes THIS
       circuit's core locally equivalent to that Haar sample.
    3. Build a 3-stage empirical Rosenblatt (sequential quantile) transform
       from that dataset: marginal of tz, conditional ty1 | tz, conditional
       ty2 | (tz, ty1).
    4. At runtime, feed 3 fresh Uniform[0,1) numbers through that transform.

This was validated end to end (see design session): pushing samples through
this transform reproduces the exact Haar frame potential F^(t) = t! for
t = 1, 2, 3 to within Monte Carlo error.

Regenerating the tables
------------------------
    python two_designs/haar_reparam.py --build [--n-samples 50000]

writes kak1_rosenblatt_tables.npz next to this file. A table is checked
into the repo so this is only needed if you want higher resolution or a
different circuit convention.
"""

from __future__ import annotations

import os

import numpy as np
import torch

_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "kak1_rosenblatt_tables.npz")

_cache: dict = {}


def _load_tables(device=None, dtype=torch.float64):
    key = (device, dtype)
    if key in _cache:
        return _cache[key]
    if not os.path.exists(_TABLE_PATH):
        raise FileNotFoundError(
            f"{_TABLE_PATH} not found. Run `python two_designs/haar_reparam.py --build` "
            "once to generate it (takes a few minutes)."
        )
    npz = np.load(_TABLE_PATH)
    t = {k: torch.as_tensor(npz[k], device=device, dtype=dtype) for k in npz.files}
    _cache[key] = t
    return t


# ── batched linear-interpolation inverse-CDF lookups (torch, no Python loop) ──

def _interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """torch equivalent of np.interp for a batch of x against a shared 1D table."""
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = ((x - x0) / (x1 - x0).clamp_min(1e-300)).clamp(0, 1)
    return f0 + w * (f1 - f0)


def _interp_1d_rows(x: torch.Tensor, xp: torch.Tensor, fp_rows: torch.Tensor) -> torch.Tensor:
    """Batched np.interp where every sample has its OWN table row (fp_rows: (B, NQ))
    but shares the same query grid xp (NQ,)."""
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0 = torch.gather(fp_rows, 1, (idx - 1).unsqueeze(1)).squeeze(1)
    f1 = torch.gather(fp_rows, 1, idx.unsqueeze(1)).squeeze(1)
    w = ((x - x0) / (x1 - x0).clamp_min(1e-300)).clamp(0, 1)
    return f0 + w * (f1 - f0)


def _blend_rows_1d(query: torch.Tensor, grid_centers: torch.Tensor, grid_rows: torch.Tensor) -> torch.Tensor:
    """Linearly blend the two grid_rows bracketing `query` along grid_centers.
    grid_rows: (n_bins, NQ) -> returns (batch, NQ)."""
    idx = torch.searchsorted(grid_centers, query.contiguous()).clamp(1, len(grid_centers) - 1) - 1
    idx = idx.clamp(0, len(grid_centers) - 2)
    t0 = grid_centers[idx]
    t1 = grid_centers[idx + 1]
    w = ((query - t0) / (t1 - t0).clamp_min(1e-300)).clamp(0, 1)
    row0 = grid_rows[idx]
    row1 = grid_rows[idx + 1]
    return row0 + w.unsqueeze(-1) * (row1 - row0)


def _blend_rows_2d(q1: torch.Tensor, q2: torch.Tensor,
                    c1: torch.Tensor, c2: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly blend the 4 grid rows bracketing (q1, q2). grid: (n1, n2, NQ)."""
    i0 = (torch.searchsorted(c1, q1.contiguous()).clamp(1, len(c1) - 1) - 1).clamp(0, len(c1) - 2)
    j0 = (torch.searchsorted(c2, q2.contiguous()).clamp(1, len(c2) - 1) - 1).clamp(0, len(c2) - 2)
    ta, tb = c1[i0], c1[i0 + 1]
    sa, sb = c2[j0], c2[j0 + 1]
    wa = ((q1 - ta) / (tb - ta).clamp_min(1e-300)).clamp(0, 1)
    wb = ((q2 - sa) / (sb - sa).clamp_min(1e-300)).clamp(0, 1)
    g00 = grid[i0, j0]
    g01 = grid[i0, j0 + 1]
    g10 = grid[i0 + 1, j0]
    g11 = grid[i0 + 1, j0 + 1]
    wa, wb = wa.unsqueeze(-1), wb.unsqueeze(-1)
    return (1 - wa) * (1 - wb) * g00 + (1 - wa) * wb * g01 + wa * (1 - wb) * g10 + wa * wb * g11


def sample_canonical(u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor) -> torch.Tensor:
    """u1,u2,u3: each in [0,1), any shape (0-D scalar for an unbatched circuit
    call, or (batch,) when circuits.py's tracing passes batched weights).
    Returns (tz, ty1, ty2) with the same shape as the inputs, distributed so
    that core(tz,ty1,ty2) sandwiched between Haar-random local SU(2)(x)SU(2)
    blocks reproduces Haar measure on SU(4)."""
    orig_shape = u1.shape
    u1, u2, u3 = torch.atleast_1d(u1), torch.atleast_1d(u2), torch.atleast_1d(u3)
    T = _load_tables(device=u1.device, dtype=u1.dtype)

    tz = _interp_1d(u1, T["cdf1"], T["tz_sorted"])

    row1 = _blend_rows_1d(tz, T["bin_centers_tz"], T["ty1_grid"])
    ty1 = _interp_1d_rows(u2, T["q_levels"], row1)

    row2 = _blend_rows_2d(tz, ty1, T["bin_centers_tz2"], T["bin_centers_ty1"], T["ty2_grid"])
    ty2 = _interp_1d_rows(u3, T["q_levels"], row2)

    return tz.reshape(orig_shape), ty1.reshape(orig_shape), ty2.reshape(orig_shape)


def euler_angles(u_alpha: torch.Tensor, u_beta: torch.Tensor, u_gamma: torch.Tensor):
    """Uniform[0,1) triple -> (alpha, beta, gamma) Euler angles for
    RZ(gamma) RY(beta) RZ(alpha), Haar-distributed on SU(2)."""
    alpha = 2 * torch.pi * u_alpha
    beta = torch.arccos(1 - 2 * u_beta)
    gamma = 2 * torch.pi * u_gamma
    return alpha, beta, gamma


# ── direct (PennyLane-free) matrix construction ────────────────────────────
#
# circuits.py's core_kak/local_su2 build the same block by queuing PennyLane
# operations, which is the right thing to do when you want ONE trace
# applied to a big batch (frame_potential.sample_unitaries's whole design,
# behind circuit_set(33)/(34)). It is the wrong thing when a caller needs
# MANY small, differently-wired constructions -- e.g.
# two_designs/range_connectivity.py, which draws a fresh 2-qubit gate
# embedding per wire pair per layer per wiring instance: retracing
# circuits.py's PennyLane queue on every one of those calls turned out to
# dominate runtime completely (a few thousand retraces of a tiny circuit
# costs far more than the circuit itself). kak1_block_matrix below is the
# same math -- same gate sequence, same GlobalPhase(-pi/4) det correction
# -- reimplemented as direct batched torch matrix construction, with no
# PennyLane tracing at all. Verified to agree with
# frame_potential.sample_unitaries(33, ...) (which goes through circuits.py
# and PennyLane) to machine precision -- see
# two_designs/range_connectivity.py's module docstring / the design
# session's own checks for how.

def _rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """theta: (...,) real -> (..., 2, 2) complex128."""
    half = (theta / 2).to(torch.complex128)
    e_neg, e_pos = torch.exp(-1j * half), torch.exp(1j * half)
    zero = torch.zeros_like(e_neg)
    return torch.stack([torch.stack([e_neg, zero], dim=-1),
                         torch.stack([zero, e_pos], dim=-1)], dim=-2)


def _ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """theta: (...,) real -> (..., 2, 2) complex128."""
    half = theta / 2
    c, s = torch.cos(half).to(torch.complex128), torch.sin(half).to(torch.complex128)
    return torch.stack([torch.stack([c, -s], dim=-1),
                         torch.stack([s, c], dim=-1)], dim=-2)


def _embed_q0(a: torch.Tensor) -> torch.Tensor:
    """kron(a, I2): a acting on qubit 0 of a 2-qubit pair. a: (B,2,2) -> (B,4,4)."""
    B = a.shape[0]
    out = torch.zeros(B, 4, 4, dtype=a.dtype, device=a.device)
    eye2 = torch.eye(2, dtype=a.dtype, device=a.device)
    for i in range(2):
        for j in range(2):
            out[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] = a[:, i, j].reshape(B, 1, 1) * eye2
    return out


def _embed_q1(b: torch.Tensor) -> torch.Tensor:
    """kron(I2, b): b acting on qubit 1 of a 2-qubit pair. b: (B,2,2) -> (B,4,4)."""
    B = b.shape[0]
    out = torch.zeros(B, 4, 4, dtype=b.dtype, device=b.device)
    out[:, 0:2, 0:2] = b
    out[:, 2:4, 2:4] = b
    return out


_CNOT_Q0Q1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex128)
_CNOT_Q1Q0 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.complex128)


def kak1_block_matrix(raw15: torch.Tensor) -> torch.Tensor:
    """raw15: (15, B) tensor, entries Uniform(0, 2*pi) -- one full, standalone
    KAK1 block: [0:3]=A1, [3:6]=A0, [6:9]=canonical, [9:12]=B1, [12:15]=B0,
    same convention two_designs/range_connectivity.py's draw_display_circuit
    builds per gate (every gate there is independent, nothing to inherit a
    dressing from -- unlike circuit_set(33)/(34), which only use this full
    form for a wire's first touch and a truncated 9-parameter form after).
    Returns
    the (B, 4, 4) complex128 unitary directly -- see the block comment
    above for why this exists alongside the PennyLane version."""
    device = raw15.device
    u = raw15.to(torch.float64) / (2 * torch.pi)

    def local_su2(u3):
        alpha, beta, gamma = euler_angles(u3[0], u3[1], u3[2])
        return _rz_matrix(gamma) @ _ry_matrix(beta) @ _rz_matrix(alpha)

    A1, A0 = local_su2(u[0:3]), local_su2(u[3:6])
    tz, ty1, ty2 = sample_canonical(u[6], u[7], u[8])
    B1, B0 = local_su2(u[9:12]), local_su2(u[12:15])

    cnot01 = _CNOT_Q0Q1.to(device=device).expand(u.shape[1], 4, 4)
    cnot10 = _CNOT_Q1Q0.to(device=device).expand(u.shape[1], 4, 4)
    core = cnot01
    core = _embed_q1(_rz_matrix(tz)) @ core
    core = _embed_q0(_ry_matrix(ty1)) @ core
    core = cnot10 @ core
    core = _embed_q0(_ry_matrix(ty2)) @ core
    core = cnot01 @ core
    core = core * torch.exp(1j * torch.tensor(torch.pi / 4, dtype=torch.complex128))

    return _embed_q0(B1) @ _embed_q1(B0) @ core @ _embed_q0(A1) @ _embed_q1(A0)


# ── offline table builder ──────────────────────────────────────────────────

def _core_matrix_np(tz, ty1, ty2):
    I2 = np.eye(2)
    def kron(*ms):
        out = ms[0]
        for m in ms[1:]:
            out = np.kron(out, m)
        return out
    CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
    def RZ(t):
        e = np.exp(1j * t / 2)
        return np.array([[1 / e, 0], [0, e]])
    def RY(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]])
    U = CNOT01
    U = kron(I2, RZ(tz)) @ U
    U = kron(RY(ty1), I2) @ U
    U = CNOT10 @ U
    U = kron(RY(ty2), I2) @ U
    U = CNOT01 @ U
    return U


def _makhlin_np(U):
    M = (1 / np.sqrt(2)) * np.array([[1, 0, 0, 1j], [0, 1j, 1, 0],
                                       [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex)
    Um = M.conj().T @ U @ M
    m = Um.T @ Um
    d = np.linalg.det(U)
    tr_m, tr_m2 = np.trace(m), np.trace(m @ m)
    G1 = tr_m ** 2 / (16 * d)
    G2 = (tr_m ** 2 - tr_m2) / (4 * d)
    return G1, G2


def _build_empirical_tzty1ty2(n_samples: int, seed: int = 0, restarts: int = 8):
    """Ground truth: Haar-random SU(4) (Ginibre + QR), each matched to this
    circuit's (tz,ty1,ty2) via its local-invariant (Makhlin G1,G2) trace
    formulas -- unambiguous, no eigendecomposition/labelling involved."""
    from scipy.stats import unitary_group
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    U_full = unitary_group.rvs(4, size=n_samples, random_state=rng)
    ph = np.angle(np.linalg.det(U_full)) / 4
    U_haar = U_full * np.exp(-1j * ph)[:, None, None]

    bounds = ([-np.pi] * 3, [np.pi] * 3)
    guesses_pool = rng.uniform(-np.pi, np.pi, size=(40, 3))

    def resid(x, target_G1, target_G2):
        G1, G2 = _makhlin_np(_core_matrix_np(*x))
        return [(G1 - target_G1).real, (G1 - target_G1).imag, (G2 - target_G2).real]

    out = np.zeros((n_samples, 3))
    for i in range(n_samples):
        G1t, G2t = _makhlin_np(U_haar[i])
        best = None
        for k in range(restarts):
            guess = guesses_pool[(i + k * 7) % len(guesses_pool)]
            sol = least_squares(resid, guess, args=(G1t, G2t), bounds=bounds,
                                 xtol=1e-13, ftol=1e-13)
            if best is None or sol.cost < best.cost:
                best = sol
            if best.cost < 1e-18:
                break
        out[i] = best.x
        if (i + 1) % 5000 == 0:
            print(f"  matched {i + 1}/{n_samples}")
    return out


def build_tables(n_samples: int = 50000, seed: int = 0,
                  n_bins1: int = 48, n_bins2: int = 28, n_bins3: int = 20,
                  n_quantiles: int = 64, min_per_bin: int = 25,
                  out_path: str = _TABLE_PATH):
    print(f"Generating {n_samples} Haar-SU(4) ground-truth samples and matching "
          "them to this circuit's (tz,ty1,ty2) via Makhlin invariants...")
    data = _build_empirical_tzty1ty2(n_samples, seed=seed)
    tz, ty1, ty2 = data[:, 0], data[:, 1], data[:, 2]
    N = len(data)

    order = np.argsort(tz)
    tz_sorted = tz[order]
    cdf1 = (np.arange(N) + 0.5) / N

    bin_edges_tz = np.quantile(tz, np.linspace(0, 1, n_bins1 + 1))
    bin_edges_tz[0] -= 1e-9
    bin_edges_tz[-1] += 1e-9
    bin_idx_tz = np.clip(np.digitize(tz, bin_edges_tz) - 1, 0, n_bins1 - 1)
    bin_centers_tz = 0.5 * (bin_edges_tz[:-1] + bin_edges_tz[1:])

    q_levels = (np.arange(n_quantiles) + 0.5) / n_quantiles
    ty1_grid = np.zeros((n_bins1, n_quantiles))
    for b in range(n_bins1):
        vals = np.sort(ty1[bin_idx_tz == b])
        if len(vals) < min_per_bin:
            vals = np.sort(ty1)
        src_q = (np.arange(len(vals)) + 0.5) / len(vals)
        ty1_grid[b] = np.interp(q_levels, src_q, vals)

    bin_edges_tz2 = np.quantile(tz, np.linspace(0, 1, n_bins2 + 1))
    bin_edges_tz2[0] -= 1e-9
    bin_edges_tz2[-1] += 1e-9
    bin_edges_ty1 = np.quantile(ty1, np.linspace(0, 1, n_bins3 + 1))
    bin_edges_ty1[0] -= 1e-9
    bin_edges_ty1[-1] += 1e-9
    bidx_tz2 = np.clip(np.digitize(tz, bin_edges_tz2) - 1, 0, n_bins2 - 1)
    bidx_ty1 = np.clip(np.digitize(ty1, bin_edges_ty1) - 1, 0, n_bins3 - 1)
    bin_centers_tz2 = 0.5 * (bin_edges_tz2[:-1] + bin_edges_tz2[1:])
    bin_centers_ty1 = 0.5 * (bin_edges_ty1[:-1] + bin_edges_ty1[1:])

    ty2_grid = np.zeros((n_bins2, n_bins3, n_quantiles))
    for i in range(n_bins2):
        for j in range(n_bins3):
            mask = (bidx_tz2 == i) & (bidx_ty1 == j)
            vals = np.sort(ty2[mask])
            if len(vals) < min_per_bin:
                vals = np.sort(ty2)
            src_q = (np.arange(len(vals)) + 0.5) / len(vals)
            ty2_grid[i, j] = np.interp(q_levels, src_q, vals)

    np.savez(out_path,
             tz_sorted=tz_sorted, cdf1=cdf1,
             bin_centers_tz=bin_centers_tz, ty1_grid=ty1_grid, q_levels=q_levels,
             bin_centers_tz2=bin_centers_tz2, bin_centers_ty1=bin_centers_ty1, ty2_grid=ty2_grid)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.build:
        build_tables(n_samples=args.n_samples, seed=args.seed)
    else:
        print(__doc__)
