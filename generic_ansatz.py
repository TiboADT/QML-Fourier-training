"""Generic, numerically-encoded ansatz builder -- a parallel path to
circuits.py's hardcoded circuit_set(1-34), for assembling new
(rotation, entanglement) combinations from a small set of axes instead of
hand-writing a new circuit_set branch per combination. Circuits 1-34 are
untouched; this is reached via architectures.Architecture(num=CODE,
generic=True) (its .build/.weight_shape/.n_trainable dispatch here instead
of to circuits.py).

Encoding
--------
A 5-digit decimal code, read most- to least-significant digit:

    [rotation][connectivity][structure][gate_class][gate]

  rotation (1st digit): how many angles per wire per rotation layer.
    1 = one axis (RY)
    2 = two axes (RX, RZ)
    3 = full SU(2) (qp.Rot -- 3 Euler angles)

  connectivity (2nd digit): 0 = none, 1 = local, 2 = all-to-all

  structure (3rd digit, only read when connectivity == 1; see
  STRUCTURE_NAMES for the paper's names for structures 1-3):
    0 = sequential              -- consecutive overlapping pairs (0,1),(1,2),...
    1 = brickwork                -- non-overlapping pairs, alternating even/odd
    2 = double brickwork        -- both the even AND odd non-overlapping
                                   pair sets applied in the same repetition
                                   (same pairs every rep -- no even/odd
                                   alternation across reps, unlike structure 1)
    3 = circular alternating    -- full-cycle cascade, alternating forward/backward
    4 = circular, single wrap   -- one full-cycle pass, including the wrap edge

  gate_class (4th digit): 0 = none, 1 = fixed, 2 = parametrized

  gate (5th digit, only read when gate_class != 0):
    fixed:        0=CNOT, 1=CZ                    (2-9 reserved -- no SWAP:
                  it permutes basis states but doesn't entangle them, so it's
                  useless as an "entangler" here, and nothing in this
                  codebase ever used it)
    parametrized: 0=CRX, 1=CRY, 2=CRZ             (3-9 reserved)
                  wires=[w1, w0] (control=w1, target=w0)

Deliberately out of scope here: composite/KAK-style entanglers, and
range-limited connectivity (needs an actual int parameter, doesn't fit a
single digit).

Design
------
Every repetition is exactly one (rotation, entanglement) pair -- there is
no separate "shift" axis and no variable-length per-repetition sequence.
What looks like "sandwich" or "interleaved" structure in circuits 5-17 is
just this same pair repeated more times, with the entangler's wiring
changing between repetitions according to `structure` (brickwork
alternates even/odd; circular alternating alternates forward/backward) --
that alternation is *part of* what the structure name means, not an
independent choice layered on top of it.

A wire is only rotated in a given repetition if that repetition's
entangling pass actually touches it. This is a no-op for every structure
except brickwork (the only one where some wires -- the boundary
ones -- sit idle on alternating repetitions): a wire idle this repetition
already carries a valid, unconsumed rotation from whenever it was last
dressed, so re-dressing it would be wasted. For 3-axis rotations this is
provably lossless (SU(2) is a group: two consecutive rotations with
nothing between them collapse to one); for 1/2-axis rotations it isn't a
strict mathematical identity, but it is what keeps a wire's rotation
count matched to how often it's actually entangled, which is the point.

There is always exactly one closing rotation layer, touching every wire,
after the last repetition's entangling gates, with nothing after it. For
connectivity == "none" this collapses the whole ansatz to a single
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
    rotation: int       # 1, 2, or 3
    connectivity: int   # 0=none, 1=local, 2=all-to-all
    structure: int      # meaningful only if connectivity == 1
    gate_class: int     # 0=none, 1=fixed, 2=parametrized
    gate: int           # meaningful only if gate_class != 0

    @property
    def is_real(self) -> bool:
        """Whether every gate in this architecture is a real matrix --
        RY rotations (a Y-rotation is real orthogonal) and a real entangler
        (none, CNOT, CZ, or CRY; not CRX/CRZ, and not RX/RZ/Rot rotations,
        all of which are complex). With RX(x) data encoding and a PauliZ
        observable, a real-valued circuit's trained output is forced to be
        an *even* function of x -- it can only fit the cosine part of a
        complex-coefficient Fourier target, no matter how well it trains."""
        if self.rotation != 1:
            return False
        if self.gate_class == 0:
            return True
        if self.gate_class == 1:
            return True  # CNOT or CZ -- both real
        return self.gate == 1  # parametrized: only CRY (gate 1) is real


_FIXED_GATES = {0: qp.CNOT, 1: qp.CZ}
_PARAM_GATES = {0: qp.CRX, 1: qp.CRY, 2: qp.CRZ}

# Paper vocabulary (Table 1) for the `structure` digit -- meaningful only
# when connectivity == 1 (local). 0 and 4 aren't in Table 1; named here for
# completeness only.
STRUCTURE_NAMES = {
    0: "sequential",
    1: "brickwork",
    2: "double brickwork",
    3: "circular alternating",
    4: "circular, single wrap",
}
CONNECTIVITY_NAMES = {0: "none", 1: "local", 2: "all-to-all"}

# Codes decode() treats as distinct but that build the exact same circuit
# as another code already in list_architectures() -- not visible from the
# digits alone, only from working out the gate algebra:
#
# structure 3 (circular alternating) and structure 4 (circular, single
# wrap) apply the same CZ gates in the same commuting order when the
# entangler is the symmetric, diagonal CZ (gate_class=1, gate=1) --
# alternating direction is only a real difference for a directional gate
# like CNOT. Keys are dropped by list_architectures(unique=True); values
# are the equivalent code that's kept.
EQUIVALENT = {11311: 11411, 21311: 21411, 31311: 31411}

# all-to-all connectivity (2) lists both (i, j) and (j, i) for every pair
# (see entangling_pairs); CZ is its own inverse, so applying it once in
# each direction per edge cancels to the identity. These three codes
# (connectivity=2, gate_class=1, gate=1, for each rotation arity) entangle
# nothing at all, duplicating the no-topology code of the same rotation
# arity (10000/20000/30000).
NO_ENTANGLEMENT = {12011, 22011, 32011}


def decode(code: int) -> GenericSpec:
    """Decode a 5-digit code into a GenericSpec, validating each digit
    against the ranges documented in this module's docstring."""
    if not (10000 <= code <= 99999):
        raise ValueError(f"generic ansatz code must be exactly 5 digits (10000-99999), got {code}")
    rotation = code // 10000
    connectivity = (code // 1000) % 10
    structure = (code // 100) % 10
    gate_class = (code // 10) % 10
    gate = code % 10

    if rotation not in (1, 2, 3):
        raise ValueError(f"rotation digit must be 1, 2, or 3, got {rotation} (code={code})")
    if connectivity not in (0, 1, 2):
        raise ValueError(f"connectivity digit must be 0, 1, or 2, got {connectivity} (code={code})")
    if connectivity == 1 and structure not in (0, 1, 2, 3, 4):
        raise ValueError(f"structure digit must be 0-4 for local connectivity, "
                          f"got {structure} (code={code})")
    if gate_class not in (0, 1, 2):
        raise ValueError(f"gate-class digit must be 0, 1, or 2, got {gate_class} (code={code})")
    if gate_class == 1 and gate not in _FIXED_GATES:
        raise ValueError(f"gate digit must be one of {sorted(_FIXED_GATES)} for a fixed "
                          f"gate, got {gate} (code={code})")
    if gate_class == 2 and gate not in _PARAM_GATES:
        raise ValueError(f"gate digit must be one of {sorted(_PARAM_GATES)} for a "
                          f"parametrized gate, got {gate} (code={code})")

    return GenericSpec(rotation, connectivity, structure, gate_class, gate)


def get_architecture_from_code(code: int) -> int:
    """The 2-digit (connectivity, structure) group a code belongs to --
    e.g. 21110 and 21111 (differing only in gate) both give 11. Collapses
    rotation arity and gate choice, isolating how *topology* alone affects
    something (see architecture_label for the readable name)."""
    return (code // 100) % 100


def architecture_label(a: int) -> str:
    """Readable name for a get_architecture_from_code group."""
    connectivity, structure = divmod(a, 10)
    if connectivity == 0:
        return "no topology"
    if connectivity == 2:
        return "all-to-all"
    return STRUCTURE_NAMES.get(structure, f"architecture {a}")


def list_architectures(unique: bool = False) -> list[int]:
    """Every generic_ansatz code, one per digit combination decode()
    accepts.

    decode() accepts (and build()/weight_tensor_shape() happily run on)
    digit values that don't actually change the resulting circuit:
    structure is only read when connectivity == 1 (local), and gate is
    only read when gate_class != 0. Enumerating every digit combination
    decode() accepts would therefore return many codes that all produce
    the exact same architecture. This instead fixes those don't-care
    digits at their single representative value (0), so every code in the
    returned list is distinct *by its digits*.

    unique=True additionally drops EQUIVALENT's keys and all of
    NO_ENTANGLEMENT -- codes that are digit-distinct but build the exact
    same circuit as another code already in the list (93 -> 87). The
    default (False, 93 codes) is what every existing sweep already uses;
    changing it would silently change what --circuits all/--generic sweeps
    without --circuits.
    """
    codes = []
    #  add the no-topology, no-gate option first (rotation-only, any rotation arity)
    for rotation in (1, 2, 3):
        codes.append(rotation * 10000)
    gate_options = [(1, t) for t in sorted(_FIXED_GATES)] + [(2, t) for t in sorted(_PARAM_GATES)]
    for rotation in (1, 2, 3):
        for connectivity in (1, 2):
            structures = (0, 1, 2, 3, 4) if connectivity == 1 else (0,)
            for structure in structures:
                for gate_class, gate in gate_options:
                    codes.append(rotation * 10000 + connectivity * 1000
                                 + structure * 100 + gate_class * 10 + gate)
    if not unique:
        return codes
    return [c for c in codes if c not in EQUIVALENT and c not in NO_ENTANGLEMENT]


# ── connectivity: which pairs are entangled at a given repetition ──────

def entangling_pairs(spec: GenericSpec, num_wires: int, rep: int) -> list[tuple[int, int]]:
    """The (w0, w1) index pairs (into 0..num_wires-1) entangled at
    repetition `rep` (0-indexed). Independent of how many repetitions
    there are in total -- callers just loop rep = 0, 1, 2, ... themselves.
    """
    if spec.connectivity == 0:
        return []

    if spec.connectivity == 2:
        # fully connected
        pairs = []
        for i in range(num_wires):
            for j in range(num_wires):
                if i != j:
                    pairs.append((i, j))
        return pairs

    # connectivity == 1 (local)
    if spec.structure == 0:  # sequential
        return [(i, i + 1) for i in range(num_wires - 1)]

    if spec.structure == 1:  # brickwork
        num_pairs = num_wires // 2
        wires_parity = 1 - (num_wires % 2)
        layer_pairs = num_pairs - rep % 2 * wires_parity
        offset = rep % 2
        return [(offset + 2 * i, offset + 2 * i + 1) for i in range(layer_pairs)]

    if spec.structure == 2:  # double brickwork (same pairs every rep -- no alternation)
        num_pairs = num_wires // 2
        wires_parity = 1 - (num_wires % 2)
        evens = [(2 * i, 2 * i + 1) for i in range(num_pairs)]
        odds = [(2 * i + 1, 2 * i + 2) for i in range(num_pairs - wires_parity)]
        return evens + odds

    if spec.structure == 3:  # circular alternating (forward/backward)
        if rep % 2 == 0:
            return [(q, (q + 1) % num_wires) for q in range(num_wires)]
        return [((q - 1) % num_wires, q) for q in range(num_wires)]

    if spec.structure == 4:  # circular, single wrap
        return [(q, (q + 1) % num_wires) for q in range(num_wires)]

    raise ValueError(f"unhandled topology structure {spec.structure}")


# ── gate application ────────────────────────────────────────────────────

def _apply_rotation(spec: GenericSpec, params_row, wire):
    """params_row: (rotation,) [+ optional trailing batch dim]."""
    if spec.rotation == 1:
        qp.RY(params_row[0], wires=wire)
    elif spec.rotation == 2:
        qp.RX(params_row[0], wires=wire)
        qp.RZ(params_row[1], wires=wire)
    else:
        qp.Rot(params_row[0], params_row[1], params_row[2], wires=wire)


def _apply_entangler(spec: GenericSpec, param, wires, swap_for_parametrized: bool):
    """wires = [w0, w1] in exactly the order entangling_pairs returned the
    pair. A fixed gate always reads it directly as [control, target].
    A parametrized gate reads [w1, w0] (control=w1, target=w0) when
    swap_for_parametrized is True; build()'s one exception is circular-
    sequential's backward pass, which passes False to keep [w0, w1]."""
    w0, w1 = wires
    if spec.gate_class == 0:
        return
    if spec.gate_class == 1:
        _FIXED_GATES[spec.gate](wires=[w0, w1])
        return
    ordered = [w1, w0] if swap_for_parametrized else [w0, w1]
    _PARAM_GATES[spec.gate](param, wires=ordered)


# ── shape / build ────────────────────────────────────────────────────────

def _layer_cost(spec: GenericSpec, num_wires: int, rep: int) -> int:
    """Exactly how many *distinct* parameters repetition `rep` reads """
    pairs = entangling_pairs(spec, num_wires, rep)
    touched = {w for pair in pairs for w in pair}
    n_rotation = len(touched) * spec.rotation
    n_entangler = len(pairs) if spec.gate_class == 2 else 0
    return n_rotation + n_entangler


def weight_tensor_shape(code: int, num_wires: int, reps: int = 1) -> tuple:
    """params shape: (reps + 1, num_wires, width)

    For all-to-all topology, width's entangler_cols = num_wires - 1: each
    wire w0 is the first element of exactly num_wires - 1 ordered pairs
    (one per other wire -- see entangling_pairs's connectivity == 2
    branch), so build()'s per-wire cp_idx_by_wire counter cycles through
    exactly 0..num_wires-2 for that wire: one write per allocated column,
    no collision and nothing left unused.
    """
    spec = decode(code)
    if spec.connectivity == 0:
        return (1, num_wires, spec.rotation)
    entangler_cols = 0
    if spec.gate_class == 2:
        entangler_cols = (num_wires - 1) if spec.connectivity == 2 else 1
    return (reps + 1, num_wires, spec.rotation + entangler_cols)


def build(code: int, params, wires=None):
    """Queue the gate sequence for `code` given a (reps + 1, num_wires,
    width) params tensor [+ optional trailing batch dim] -- 
    """
    spec = decode(code)
    if wires is None:
        raise ValueError("the generic ansatz needs `wires` given explicitly")
    num_wires = len(wires)
    reps = params.shape[0] - 1

    for rep in range(reps):
        pairs = entangling_pairs(spec, num_wires, rep)
        touched = sorted({w for pair in pairs for w in pair})
        for w in touched:
            _apply_rotation(spec, params[rep, w, :spec.rotation], wires[w])
        # see _apply_entangler's docstring: circular-sequential's backward
        # pass (structure 3, odd rep) is the one case that does NOT swap
        # which wire is control vs target.
        swap = not (spec.structure == 3 and rep % 2 == 1)
        # each pair's parameter lives at its first wire's entangler
        # column(s) -- cp_idx_by_wire counts up per wire so all-to-all's
        # num_wires-1 pairs per wire land on distinct columns (see
        # weight_tensor_shape's docstring for the no-collision argument).
        cp_idx_by_wire = {}
        for (w0, w1) in pairs:
            cp_idx = cp_idx_by_wire.get(w0, 0)
            cp_idx_by_wire[w0] = cp_idx + 1
            if spec.gate_class == 2:
                param = params[rep, w0, spec.rotation + cp_idx]
            else:
                param = None
            _apply_entangler(spec, param, [wires[w0], wires[w1]], swap)

    # closing layer: always the last slice, rotation-only
    for w in range(num_wires):
        _apply_rotation(spec, params[reps, w, :spec.rotation], wires[w])


def n_trainable(code: int, num_wires: int, reps: int = 1) -> int:
    """Exact count of weight-tensor entries build() actually reads"""
    spec = decode(code)
    total = num_wires * spec.rotation  # the closing layer
    for rep in range(reps):
        total += _layer_cost(spec, num_wires, rep)
    return total


# ── correspondence with circuits.py's hardcoded circuit_set(1-19) ──────────

@dataclass(frozen=True)
class LegacyCorrespondence:
    """How circuit_set(num) (num in 1-19) relates to generic_ansatz(code).

    rep_multiplier: generic's own `reps` must be `rep_multiplier *
    circuit_set(num)'s reps` for the correspondence to hold -- only ever 1 or
    2. It's 2 for the structures whose circuit_set loop does a full forward+
    backward (circular alternating: 13/14/15) or even+odd (brickwork:
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
