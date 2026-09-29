"""Local random circuits (two_designs/haar_reparam.py's exact-Haar KAK1
block, brickwork-tiled -- circuits.py's circuit 34) versus "permuted
brickwork" -- the same blocks, but each layer wires up a *fresh random*
pairing of qubits instead of a fixed nearest-neighbour pattern ("Family C"
in the "Building 2-Designs" design notes). The claim worth checking: at
matched two-qubit gate count, letting gates reach further needs less depth
to approach a 2-design -- connectivity as its own axis, independent of
gate count.

This module makes "how far a gate is allowed to reach" a tunable integer,
`max_range`: on qubits laid out on a line 0..n_qubits-1, a layer may only
pair qubits i, j with |i - j| <= max_range.

    max_range = 1              only nearest-neighbour pairs -- the same
                                *style* of connectivity as circuit 34,
                                though each layer's matching is
                                independently randomized here rather than
                                circuit 34's fixed alternating-parity
                                pattern (see random_matching's docstring)
    max_range >= n_qubits - 1  every pair is reachable -- "permuted
                                brickwork" / the fully-connected limit,
                                no restriction at all
    1 < max_range < n_qubits-1 the graduated middle ground the design
                                notes don't explore but this module lets
                                you sweep

Why a new sampler rather than a circuit_set number: the wiring genuinely
has to be random *per sample*, not just per parameter -- a circuit_set
architecture is one fixed gate sequence traced once and then batch-applied
with random angles (see frame_potential.sample_unitaries), which has no
room for "which wires" to vary within a batch. So this plugs into
frame_potential's generic sampler interface instead (same reason
two_designs/clifford_group.py does) -- but it still reuses circuit 33's
math for the actual 2-qubit gate (via haar_reparam.kak1_block_matrix, a
direct/PennyLane-free reimplementation used here specifically because this
module needs MANY small, differently-wired constructions -- retracing
circuits.py's PennyLane queue that many times turned out to dominate
runtime completely; see that function's docstring) and
frame_potential.apply_embedded_gate for embedding it at a chosen wire
pair. This module only adds the wiring logic circuit_set has no room for.

Two-level sampling, matching the design notes' own methodology ("Permuted
values averaged over 40 wiring samples"): each requested batch is built
from several independent wiring draws (a fresh random matching per layer),
each evaluated at `samples_per_wiring` independent angle-draws before
moving to the next wiring. Within one wiring's sub-batch, samples share
wiring but not angles -- some of the sub-batch's own internal variance
comes from a hidden shared variable (the wiring), which the usual pairwise
variance estimate doesn't model, so prefer more, smaller wiring draws over
few, large ones when precision matters (samples_per_wiring's docstring
below).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from circuits import core_kak, local_su2
from frame_potential import apply_embedded_gate, get_device
from two_designs.haar_reparam import euler_angles, kak1_block_matrix, sample_canonical


def random_matching(n_qubits: int, max_range: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """One layer's wiring: a greedy random matching on the "range graph"
    (qubits i, j connected iff |i - j| <= max_range). Shuffle qubit order,
    then walk it pairing each still-unmatched qubit with the first
    still-unmatched, in-range qubit later in the shuffle. Always valid (no
    qubit reused); not necessarily maximum (some qubits can be left
    unpaired this layer even when a valid pairing for them exists) or
    uniform over all matchings -- a simple, well-defined randomization is
    all this needs, not a combinatorially exact sampler.

    max_range >= n_qubits - 1 (no restriction) reduces this to a standard
    random-shuffle-and-pair construction, which *is* uniform over perfect
    matchings.
    """
    order = rng.permutation(n_qubits)
    matched = np.zeros(n_qubits, dtype=bool)
    pairs = []
    for idx in range(n_qubits):
        i = int(order[idx])
        if matched[i]:
            continue
        for jdx in range(idx + 1, n_qubits):
            j = int(order[jdx])
            if not matched[j] and abs(i - j) <= max_range:
                pairs.append((i, j) if i < j else (j, i))
                matched[i] = matched[j] = True
                break
    return pairs


def sample_range_connected_unitaries(n_qubits: int, reps: int, max_range: int, batch_size: int, *,
                                      device: Optional[torch.device] = None,
                                      dtype: torch.dtype = torch.complex64,
                                      generator: Optional[torch.Generator] = None,
                                      samples_per_wiring: int = 50) -> torch.Tensor:
    """batch_size independent samples of a `reps`-layer circuit where every
    layer applies circuit 33 (the exact-Haar KAK1 block) to a fresh
    random_matching restricted to `max_range`. Matches the generic
    frame_potential sampler interface (batch_size, *, device, dtype,
    generator) -> Tensor[batch_size, d, d].

    samples_per_wiring: batch_size is split into ceil(batch_size /
    samples_per_wiring) independent wiring draws, each evaluated at up to
    samples_per_wiring independent angle-draws -- see the module docstring
    for why this two-level structure exists. Smaller values give more
    wiring diversity per call (closer to the "wiring genuinely varies per
    sample" ideal) at the cost of more, smaller batched-gate-application
    calls; larger values are faster but understate the ensemble's true
    variance if pushed too far (an extreme, all_in_one_wiring, would
    silently measure "many angle-draws on ONE fixed wiring", not the
    intended ensemble at all).

    Reproducible via `generator` (a torch.Generator) despite the wiring
    being drawn with numpy: a seed is derived from `generator` once per
    call, unlike two_designs/clifford_group.py's stim-backed sampler,
    which has no seed hook at all.
    """
    if device is None:
        device = get_device()
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()) if generator is not None else None
    rng = np.random.default_rng(seed)

    d = 2 ** n_qubits
    chunks = []
    remaining = batch_size
    while remaining > 0:
        b = min(samples_per_wiring, remaining)
        U = torch.eye(d, dtype=torch.complex128, device=device).expand(b, d, d).clone()
        for _ in range(reps):
            for (w0, w1) in random_matching(n_qubits, max_range, rng):
                raw15 = 2 * torch.pi * torch.rand(15, b, dtype=torch.float64, generator=generator)
                G = kak1_block_matrix(raw15.to(device))
                U = apply_embedded_gate(U, G, (w0, w1), n_qubits)
        chunks.append(U)
        remaining -= b
    return torch.cat(chunks, dim=0).to(dtype=dtype)


def draw_display_circuit(n_qubits: int, reps: int, max_range: int, *, seed: Optional[int] = None):
    """One concrete, PennyLane-traceable realization of this ensemble --
    display/inspection only (checks/show.py), never called from
    sample_range_connected_unitaries above, which deliberately avoids
    PennyLane tracing entirely (see the module docstring: retracing it
    per sample was the original performance bug). Draws its own wiring
    (random_matching, one fresh draw per layer) and its own random gate
    angles, both seeded from `seed` if given.

    Returns (circuit_fn, wiring): circuit_fn takes no arguments and queues
    directly inside a qml.qnode; wiring is the list of per-layer pair lists
    actually drawn, for printing alongside the diagram (a qml.draw ASCII
    diagram alone doesn't make the *reach* of each layer easy to eyeball).
    """
    rng = np.random.default_rng(seed)
    wiring = [random_matching(n_qubits, max_range, rng) for _ in range(reps)]
    gen = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 31 - 1)))

    def circuit_fn():
        for layer_pairs in wiring:
            for (w0, w1) in layer_pairs:
                raw15 = 2 * torch.pi * torch.rand(15, generator=gen, dtype=torch.float64)
                u = raw15 / (2 * torch.pi)
                # a full, standalone Haar-exact block -- every pair here is
                # independent by design (see module docstring), so unlike
                # circuits.py's circuits 33/34 there's no earlier gate on
                # either wire to inherit a dressing from.
                local_su2(*euler_angles(u[0], u[1], u[2]), w0)
                local_su2(*euler_angles(u[3], u[4], u[5]), w1)
                core_kak(*sample_canonical(u[6], u[7], u[8]), [w0, w1])
                local_su2(*euler_angles(u[9], u[10], u[11]), w0)
                local_su2(*euler_angles(u[12], u[13], u[14]), w1)

    return circuit_fn, wiring
