"""Generic, numerically-encoded ansatz builder -- a parallel path to
circuits.py's hardcoded circuit_set(1-34), for assembling new
(rotation, entanglement) combinations from a small set of axes instead of
hand-writing a new circuit_set branch per combination. Circuits 1-34 are
untouched; this is reached via circuit_set(num=CODE, generic=True) (and
the matching weight_tensor_shape(..., generic=True) / n_trainable(...,
generic=True)).

Encoding
--------
A 5-digit decimal code, read most- to least-significant digit:

    [rotation][topo category][topo structure][gate category][gate type]

  rotation (1st digit): how many angles per wire per rotation layer.
    1 = one axis (RY)
    2 = two axes (RX, RZ)
    3 = full SU(2) (qp.Rot -- 3 Euler angles)

  topo category (2nd digit): 0 = none, 1 = local, 2 = all-to-all

  topo structure (3rd digit, only read when topo category == 1):
    0 = chain, sequential       -- consecutive overlapping pairs (0,1),(1,2),...
    1 = chain, brickwall        -- non-overlapping pairs, alternating even/odd
    2 = chain, brickwall, 2 bricks per layer -- both the even AND odd
                                   non-overlapping pair sets applied in the
                                   same repetition (same pairs every rep --
                                   no even/odd alternation across reps,
                                   unlike structure 1)
    3 = circular, sequential    -- full-cycle cascade, alternating forward/backward
    4 = circular, single wrap   -- one full-cycle pass, including the wrap edge

  gate category (4th digit): 0 = none, 1 = fixed, 2 = parametrized

  gate type (5th digit, only read when gate category != 0):
    fixed:        0=CNOT, 1=CZ                    (2-9 reserved -- no SWAP:
                  it permutes basis states but doesn't entangle them, so it's
                  useless as an "entangler" here, and nothing in this
                  codebase ever used it)
    parametrized: 0=CRX, 1=CRY, 2=CRZ             (3-9 reserved)
                  wires=[w1, w0] (control=w1, target=w0) -- matches
                  circuits.py's circuits 4/6/8/14/17/19, which now use
                  CRX directly instead of the Hadamard-ControlledPhaseShift-
                  Hadamard sandwich those used to be written as (that
                  sandwich is closely related to CRX, not identical to it:
                  H.CP(theta).H == CRX(theta) . PhaseShift(theta/2,
                  control) -- using CRX directly here is a deliberate
                  simplification, not a reproduction of the old sandwich).
                  No ControlledPhaseShift option: circuits.py's former CP
                  circuits (3/5/7/13/16/18) now use CRZ directly for the
                  same reason (CP(theta) == CRZ(theta) . PhaseShift(theta/2,
                  control) -- closely related, not identical), so CP
                  itself appears nowhere in this codebase any more.

Deliberately out of scope here (per design discussion): composite/KAK-style
entanglers, and range-limited connectivity (needs an actual int parameter,
doesn't fit a single digit).

Design
------
Every repetition is exactly one (rotation, entanglement) pair -- there is
no separate "shift" axis and no variable-length per-repetition sequence.
What looks like "sandwich" or "interleaved" structure in circuits 5-17 is
just this same pair repeated more times, with the entangler's wiring
changing between repetitions according to `structure` (brickwall
alternates even/odd; circular-sequential alternates forward/backward) --
that alternation is *part of* what the structure name means, not an
independent choice layered on top of it.

A wire is only rotated in a given repetition if that repetition's
entangling pass actually touches it. This is a no-op for every structure
except chain-brickwall (the only one where some wires -- the boundary
ones -- sit idle on alternating repetitions): a wire idle this repetition
already carries a valid, unconsumed rotation from whenever it was last
dressed, so re-dressing it would be wasted. For 3-axis rotations this is
provably lossless (SU(2) is a group: two consecutive rotations with
nothing between them collapse to one); for 1/2-axis rotations it isn't a
strict mathematical identity, but it is what keeps a wire's rotation
count matched to how often it's actually entangled, which is the point.

There is always exactly one closing rotation layer, touching every wire,
after the last repetition's entangling gates, with nothing after it. For
topo category == "none" this collapses the whole ansatz to a single
rotation layer regardless of how many repetitions are requested -- the
same rule, no special-casing, since there's never an entangling pass to
qualify any wire in the repeated part.
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qp


# ── decoding ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenericSpec:
    rotation_arity: int    # 1, 2, or 3
    topo_category: int     # 0=none, 1=local, 2=all-to-all
    topo_structure: int    # meaningful only if topo_category == 1
    gate_category: int     # 0=none, 1=fixed, 2=parametrized
    gate_type: int         # meaningful only if gate_category != 0


_FIXED_GATES = {0: qp.CNOT, 1: qp.CZ}
_PARAM_GATES = {0: qp.CRX, 1: qp.CRY, 2: qp.CRZ}


def decode(code: int) -> GenericSpec:
    """Decode a 5-digit code into a GenericSpec, validating each digit
    against the ranges documented in this module's docstring."""
    if not (10000 <= code <= 99999):
        raise ValueError(f"generic ansatz code must be exactly 5 digits (10000-99999), got {code}")
    rotation_arity = code // 10000
    topo_category = (code // 1000) % 10
    topo_structure = (code // 100) % 10
    gate_category = (code // 10) % 10
    gate_type = code % 10

    if rotation_arity not in (1, 2, 3):
        raise ValueError(f"rotation digit must be 1, 2, or 3, got {rotation_arity} (code={code})")
    if topo_category not in (0, 1, 2):
        raise ValueError(f"topology-category digit must be 0, 1, or 2, got {topo_category} (code={code})")
    if topo_category == 1 and topo_structure not in (0, 1, 2, 3, 4):
        raise ValueError(f"topology-structure digit must be 0-4 for local topology, "
                          f"got {topo_structure} (code={code})")
    if gate_category not in (0, 1, 2):
        raise ValueError(f"gate-category digit must be 0, 1, or 2, got {gate_category} (code={code})")
    if gate_category == 1 and gate_type not in _FIXED_GATES:
        raise ValueError(f"gate-type digit must be one of {sorted(_FIXED_GATES)} for a fixed "
                          f"gate, got {gate_type} (code={code})")
    if gate_category == 2 and gate_type not in _PARAM_GATES:
        raise ValueError(f"gate-type digit must be one of {sorted(_PARAM_GATES)} for a "
                          f"parametrized gate, got {gate_type} (code={code})")

    return GenericSpec(rotation_arity, topo_category, topo_structure, gate_category, gate_type)


def list_architectures() -> list[int]:
    """Every distinct, non-redundant generic_ansatz code.

    decode() accepts (and build()/weight_tensor_shape() happily run on)
    digit values that don't actually change the resulting circuit:
    topo_structure is only read when topo_category == 1 (local), and
    gate_type is only read when gate_category != 0. Enumerating every
    digit combination decode() accepts would therefore return many codes
    that all produce the exact same architecture. This instead fixes
    those don't-care digits at their single representative value (0),
    so every code in the returned list is architecturally distinct from
    every other one.
    """
    codes = []
    #  add the no-topology, no-gate option first (rotation-only, any rotation_arity)
    for rotation in (1, 2, 3):
        codes.append(rotation * 10000)
    gate_options = [(1, t) for t in sorted(_FIXED_GATES)] + [(2, t) for t in sorted(_PARAM_GATES)]
    for rotation in (1, 2, 3):
        for topo_category in (1, 2):
            topo_structures = (0, 1, 2, 3, 4) if topo_category == 1 else (0,)
            for topo_structure in topo_structures:
                for gate_category, gate_type in gate_options:
                    codes.append(rotation * 10000 + topo_category * 1000
                                 + topo_structure * 100 + gate_category * 10 + gate_type)
    return codes


# ── connectivity: which pairs are entangled at a given repetition ──────

def entangling_pairs(spec: GenericSpec, num_wires: int, rep: int) -> list[tuple[int, int]]:
    """The (w0, w1) index pairs (into 0..num_wires-1) entangled at
    repetition `rep` (0-indexed). Independent of how many repetitions
    there are in total -- callers just loop rep = 0, 1, 2, ... themselves.
    """
    if spec.topo_category == 0:
        return []

    if spec.topo_category == 2:
        # fully connected
        pairs = []
        for i in range(num_wires):
            for j in range(num_wires):
                if i != j:
                    pairs.append((i, j))
        return pairs

    # topo_category == 1 (local)
    if spec.topo_structure == 0:  # chain, sequential
        return [(i, i + 1) for i in range(num_wires - 1)]

    if spec.topo_structure == 1:  # chain, brickwall
        num_pairs = num_wires // 2
        wires_parity = 1 - (num_wires % 2)
        layer_pairs = num_pairs - rep % 2 * wires_parity
        offset = rep % 2
        return [(offset + 2 * i, offset + 2 * i + 1) for i in range(layer_pairs)]

    if spec.topo_structure == 2:  # chain, brickwall, 2 bricks per layer (same pairs every rep -- no alternation)
        num_pairs = num_wires // 2
        wires_parity = 1 - (num_wires % 2)
        evens = [(2 * i, 2 * i + 1) for i in range(num_pairs)]
        odds = [(2 * i + 1, 2 * i + 2) for i in range(num_pairs - wires_parity)]
        return evens + odds

    if spec.topo_structure == 3:  # circular, sequential (alternating forward/backward)
        if rep % 2 == 0:
            return [(q, (q + 1) % num_wires) for q in range(num_wires)]
        return [((q - 1) % num_wires, q) for q in range(num_wires)]

    if spec.topo_structure == 4:  # circular, single wrap
        return [(q, (q + 1) % num_wires) for q in range(num_wires)]

    raise ValueError(f"unhandled topology structure {spec.topo_structure}")


# ── gate application ────────────────────────────────────────────────────

def _apply_rotation(spec: GenericSpec, params_row, wire):
    """params_row: (rotation_arity,) [+ optional trailing batch dim]."""
    if spec.rotation_arity == 1:
        qp.RY(params_row[0], wires=wire)
    elif spec.rotation_arity == 2:
        qp.RX(params_row[0], wires=wire)
        qp.RZ(params_row[1], wires=wire)
    else:
        qp.Rot(params_row[0], params_row[1], params_row[2], wires=wire)


def _apply_entangler(spec: GenericSpec, param, wires, swap_for_parametrized: bool):
    """wires = [w0, w1] in exactly the order entangling_pairs returned the
    pair. A fixed gate always reads it directly as [control, target] --
    that convention is what matches circuits.py's CNOT/CZ circuits
    (2/9/10/11/12/15/32) for every structure, including circular-
    sequential's backward pass. A parametrized gate (CRX/CRY/CRZ) usually
    needs it reversed, [w1, w0] = [control, target] -- matching circuits
    4/6/8/14/17/19 -- *except* on circular-sequential's (structure 3)
    backward pass, where circuits.py's own circuit 14 happens to use the
    direct, un-reversed order instead (an inherited asymmetry between
    circuit 13/14/15's forward and backward loops, not a choice made
    here). Confirmed by reusing circuit 14's own params tensor directly
    and finding the reversed convention gave the wrong unitary
    specifically on that pass. `swap_for_parametrized` is build()'s
    per-rep flag for which applies.
    """
    w0, w1 = wires
    if spec.gate_category == 0:
        return
    if spec.gate_category == 1:
        _FIXED_GATES[spec.gate_type](wires=[w0, w1])
        return
    ordered = [w1, w0] if swap_for_parametrized else [w0, w1]
    _PARAM_GATES[spec.gate_type](param, wires=ordered)


# ── shape / build ────────────────────────────────────────────────────────

def _layer_cost(spec: GenericSpec, num_wires: int, rep: int) -> int:
    """Exactly how many flat parameter entries repetition `rep` reads:
    rotation_arity per wire actually touched this rep (not num_wires --
    chain-brickwall's idle boundary wires read none), plus exactly one
    entangler parameter per pair (not per wire: all-to-all has up to
    num_wires-1 pairs sharing a wire, and each needs its own parameter,
    never a shared one)."""
    pairs = entangling_pairs(spec, num_wires, rep)
    touched = {w for pair in pairs for w in pair}
    n_rotation = len(touched) * spec.rotation_arity
    n_entangler = len(pairs) if spec.gate_category == 2 else 0
    return n_rotation + n_entangler


def weight_tensor_shape(code: int, num_wires: int, reps: int = 1) -> tuple:
    """params shape: (total,), a flat vector sized to exactly what `reps`
    repetitions plus one closing rotation-only layer actually read -- see
    _layer_cost. No wire or pair ever shares a slot with another, and no
    slot goes allocated-but-unread the way a rectangular (reps, num_wires,
    width) tensor would for chain-brickwall's idle wires or all-to-all's
    per-wire-shared columns.
    """
    spec = decode(code)
    total = num_wires * spec.rotation_arity  # the closing layer
    for rep in range(reps):
        total += _layer_cost(spec, num_wires, rep)
    return (total,)


def build(code: int, params, wires=None):
    """Queue the gate sequence for `code` given a flat params tensor
    [+ optional trailing batch dim] -- see weight_tensor_shape. `wires`
    must be given explicitly (there's nothing to default it from once
    num_wires alone doesn't determine wire *labels*, matching
    circuit_set(34)'s own convention). `reps` itself is recovered by
    consuming the flat tensor repetition by repetition until only the
    closing layer's worth is left, the same convention circuit_set(34)
    uses for the same reason (a flat tensor's own length doesn't carry
    `reps` separately).
    """
    spec = decode(code)
    if wires is None:
        raise ValueError("the generic ansatz needs `wires` given explicitly")
    num_wires = len(wires)
    closing_cost = num_wires * spec.rotation_arity
    remaining = params.shape[0] - closing_cost
    offset = 0
    rep = 0
    while remaining > 0:
        pairs = entangling_pairs(spec, num_wires, rep)
        cost = _layer_cost(spec, num_wires, rep)
        if cost == 0:
            break  # topo_category=0: nothing left to consume in the repeated part, ever
        touched = sorted({w for pair in pairs for w in pair})
        for w in touched:
            _apply_rotation(spec, params[offset:offset + spec.rotation_arity], wires[w])
            offset += spec.rotation_arity
        # see _apply_entangler's docstring: circular-sequential's backward
        # pass (structure 3, odd rep) is the one case that does NOT swap
        # which wire is control vs target.
        swap = not (spec.topo_structure == 3 and rep % 2 == 1)
        for (w0, w1) in pairs:
            if spec.gate_category == 2:
                param = params[offset]
                offset += 1
            else:
                param = None
            _apply_entangler(spec, param, [wires[w0], wires[w1]], swap)
        remaining -= cost
        rep += 1

    for w in range(num_wires):
        _apply_rotation(spec, params[offset:offset + spec.rotation_arity], wires[w])
        offset += spec.rotation_arity


# ── correspondence with circuits.py's hardcoded circuit_set(1-19) ──────────

@dataclass(frozen=True)
class LegacyCorrespondence:
    """How circuit_set(num) (num in 1-19) relates to generic_ansatz(code).

    rep_multiplier: generic's own `reps` must be `rep_multiplier *
    circuit_set(num)'s reps` for the correspondence to hold -- only ever 1 or
    2. It's 2 for the structures whose circuit_set loop does a full forward+
    backward (circular-sequential: 13/14/15) or even+odd (plain brickwall:
    11/12) round per repetition, while generic_ansatz's own `reps` counts
    each half separately.

    gap: what, if anything, generic_ansatz(code) has that circuit_set(num)
    doesn't. 'none' -- byte-identical. 'closing'/'leading' -- generic has
    exactly one extra rotation layer, trailing or leading (the mandatory
    layer every generic_ansatz circuit has, that these particular legacy
    circuits don't). Anything else is a free-text description of a real
    structural difference beyond just an extra layer.
    """
    code: int
    rep_multiplier: int = 1
    gap: str = "closing"
    note: str = ""


LEGACY_CIRCUIT_CODES: dict[int, LegacyCorrespondence] = {
    1: LegacyCorrespondence(
        20000, gap="none",
        note="reps=1 only -- generic collapses any repeat count to a single "
             "rotation layer; circuit_set(1) repeats independent RX-RZ "
             "blocks for reps>1, which generic can't produce"),
    2: LegacyCorrespondence(21010),
    3: LegacyCorrespondence(21022),
    4: LegacyCorrespondence(21020),
    5: LegacyCorrespondence(
        22022, gap="none",
        note="reps=1 only -- circuit_set(5) repeats the whole "
             "rotate-entangle-rotate sandwich independently for reps>1, "
             "which generic can't produce (same limitation as circuit 1)"),
    6: LegacyCorrespondence(
        22020, gap="none", note="reps=1 only -- same as circuit 5"),
    7: LegacyCorrespondence(
        21122,
        gap="full-width rotation between the even and odd entangling "
            "passes (no idle-skip) -- not just the closing layer",
        note="closest topology/gate match; a genuine structural gap remains"),
    8: LegacyCorrespondence(
        21120,
        gap="full-width rotation between the even and odd entangling "
            "passes (no idle-skip) -- not just the closing layer",
        note="closest topology/gate match; same gap as circuit 7"),
    9: LegacyCorrespondence(
        11011,
        gap="leading Hadamard has no generic equivalent; closing rotation "
            "is RX, generic's 1-axis option is RY",
        note="topology reference only -- not representable"),
    10: LegacyCorrespondence(11411, gap="leading"),
    11: LegacyCorrespondence(
        21110, rep_multiplier=2,
        note="exact topology match only for num_wires even -- odd "
             "num_wires adds an idle-skip mismatch. Rotation axis RY+RZ "
             "vs generic's RX+RZ accepted as equivalent"),
    12: LegacyCorrespondence(
        21111, rep_multiplier=2, note="same as circuit 11"),
    13: LegacyCorrespondence(11322, rep_multiplier=2),
    14: LegacyCorrespondence(11320, rep_multiplier=2),
    15: LegacyCorrespondence(11310, rep_multiplier=2),
    16: LegacyCorrespondence(21222),
    17: LegacyCorrespondence(21220),
    18: LegacyCorrespondence(21422),
    19: LegacyCorrespondence(21420),
}
