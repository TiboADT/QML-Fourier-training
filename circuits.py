import pennylane as qp
import torch

import haar_reparam

def full_SU2(params, wires = None):
    """ 
    params : tenor of shape ( num_wires, 3)
    """
    if wires is None:
        wires = list(range(params.shape[0]))
    num_wires = params.shape[0]
    for i in range(num_wires):
        qp.Rot(*params[i], wires=wires[i])

def two_rotations(params, wires = None):
    """ 
    params : tenor of shape (num_wires, 2)
    """
    num_wires = params.shape[0]
    if wires is None:
        wires = list(range(num_wires))
    for i in range(num_wires):
        qp.RY(params[i, 0], wires=wires[i])
        qp.RX(params[i, 1], wires=wires[i])

def CNOT_cascade(wires , circular = False, ascending = False):
    """ 
    wires : list of wires to apply the CNOT cascade on
    circular : if True, apply a CNOT from the last wire to the first wire
    ascending : if True, apply the CNOTs in ascending order, else in descending order
    """
    if ascending:
        for i in range(len(wires) - 1):
            qp.CNOT(wires=[wires[i], wires[i + 1]])
        if circular:
            qp.CNOT(wires=[wires[-1], wires[0]])
    else:
        for i in range(len(wires) - 1, 0, -1):
            qp.CNOT(wires=[wires[i], wires[i - 1]])
        if circular:
            qp.CNOT(wires=[wires[0], wires[-1]])


def rx_rz_layer(params, wires):
    """Apply RX and RZ to each wire. params shape: (num_wires, 2)"""
    for i, w in enumerate(wires):
        qp.RX(params[i, 0], wires=w)
        qp.RZ(params[i, 1], wires=w)


def parametrized_CZ_chain(params, wires):
    """Apply parametrized CZ (CP) on nearest-neighbor pairs. params shape: (num_wires-1,)"""
    for i in range(len(wires) - 1):
        qp.ControlledPhaseShift(params[i], wires=[wires[i], wires[i + 1]])


def parametrized_CX_chain(params, wires):
    """Apply parametrized CX (H-CP-H) on nearest-neighbor pairs. params shape: (num_wires-1,)"""
    for i in range(len(wires) - 1):
        qp.Hadamard(wires=wires[i])
        qp.ControlledPhaseShift(params[i], wires=[wires[i], wires[i + 1]])
        qp.Hadamard(wires=wires[i])


def kak1_core(tz, ty1, ty2, wires):
    """The 3-CNOT canonical (non-local) core of Tucci's KAK1: realizes
    exp(i(k1 XX + k2 YY + k3 ZZ)) for (k1,k2,k3) a fixed linear function of
    (tz,ty1,ty2). wires: [w0, w1]. See haar_reparam.py for how (tz,ty1,ty2)
    must be distributed for the *dressed* (locals-on-both-sides) gate to be
    Haar-random on SU(4).

    Each CNOT has determinant -1, so 3 of them make the raw circuit land in
    the det=-1 sheet of U(4), not SU(4) -- a fixed GlobalPhase(-pi/4) cancels
    that (det scales by exp(-i*4*phi) for a 2-qubit global phase), landing
    exactly on SU(4) as KAK1 requires.
    """
    w0, w1 = wires
    qp.CNOT(wires=[w0, w1])
    qp.RZ(tz, wires=w1)
    qp.RY(ty1, wires=w0)
    qp.CNOT(wires=[w1, w0])
    qp.RY(ty2, wires=w0)
    qp.CNOT(wires=[w0, w1])
    qp.GlobalPhase(-torch.pi / 4, wires=wires)


def kak1_local_su2(u, wire):
    """u : tensor of shape (3, ...), entries Uniform[0,1) -> Haar-random SU(2)
    on `wire` (RZ(gamma) RY(beta) RZ(alpha), see haar_reparam.euler_angles)."""
    alpha, beta, gamma = haar_reparam.euler_angles(u[0], u[1], u[2])
    qp.RZ(alpha, wires=wire)
    qp.RY(beta, wires=wire)
    qp.RZ(gamma, wires=wire)


def kak1_local_su2_naive(raw3, wire):
    """raw3 : tensor of shape (3, ...), entries Uniform(0, 2*pi) used
    directly as RZ-RY-RZ Euler angles -- NOT Haar-random on SU(2) (the
    middle angle needs haar_reparam.euler_angles' arccos correction for
    that; used bare here it over-samples the poles of the Bloch sphere
    relative to the equator). Exists as the "no reparametrization" half of
    the kak1_block_naive / kak1_haar_block ablation pair, circuits 33-36."""
    qp.RZ(raw3[0], wires=wire)
    qp.RY(raw3[1], wires=wire)
    qp.RZ(raw3[2], wires=wire)


def kak1_block_naive(raw15, wires):
    """Same gate structure as kak1_haar_block (4 local SU(2) blocks around
    the 3-CNOT canonical core, circuits 33/34) but with the raw
    Uniform(0, 2*pi) parameters used directly as gate angles everywhere --
    i.e. no reparametrization at all, neither the closed-form local
    (Bloch-sphere) correction nor haar_reparam.sample_canonical's
    Rosenblatt transform for the 3 non-local angles. Backs circuits 35/36,
    which exist purely so frame_potential can quantify what the
    reparametrization in 33/34 buys you -- same circuit, same parameter
    count, only the sampling distribution differs.

    raw15 layout: same as kak1_haar_block: [0:3]=A1, [3:6]=A0,
    [6:9]=canonical (tz,ty1,ty2 used directly, unlike kak1_haar_block),
    [9:12]=B1, [12:15]=B0.
    """
    w0, w1 = wires
    kak1_local_su2_naive(raw15[0:3], w0)
    kak1_local_su2_naive(raw15[3:6], w1)
    kak1_core(raw15[6], raw15[7], raw15[8], [w0, w1])
    kak1_local_su2_naive(raw15[9:12], w0)
    kak1_local_su2_naive(raw15[12:15], w1)


def kak1_haar_block(raw15, wires):
    """raw15 : tensor of shape (15, ...), entries Uniform(0, 2*pi) -- exactly
    what sample_unitaries/circuit_set already generate for every circuit in
    this file. wires: [w0, w1].

    Realizes U = (A1 (x) A0) exp(i(k1 XX + k2 YY + k3 ZZ)) (B1 (x) B0)
    (Tucci's KAK1, arXiv:quant-ph/0507171 Eq. 1) with U exactly
    Haar-distributed on SU(4) -- see haar_reparam.py.

    raw15 layout: [0:3]=A1, [3:6]=A0, [6:9]=canonical (u1,u2,u3), [9:12]=B1, [12:15]=B0.
    """
    w0, w1 = wires
    u = raw15 / (2 * torch.pi)
    kak1_local_su2(u[0:3], w0)   # A1
    kak1_local_su2(u[3:6], w1)   # A0
    tz, ty1, ty2 = haar_reparam.sample_canonical(u[6], u[7], u[8])
    kak1_core(tz, ty1, ty2, [w0, w1])
    kak1_local_su2(u[9:12], w0)  # B1
    kak1_local_su2(u[12:15], w1)  # B0


def circuit_set(name: str = None, num: int = None):
    """Return a function corresponding to the named or numbered circuit.

    Circuits 1–19 are direct translations of the Qiskit circuit_set circuits.
    """
    
    circuit_names = {
        "SU4": 31,
        "Brickwall": 32,
        "StronglyEntangling": 30,
        "KAK1_Haar": 33,
        "KAK1_Haar_Brickwall": 34,
        "KAK1_Uniform": 35,
        "KAK1_Uniform_Brickwall": 36,
    }
    # extend as needed

    if num is None:
        if name is None:
            raise ValueError("You must provide either a circuit name or a circuit number.")
        num = circuit_names[name]

    # ------------------------------------------------------------------
    # Circuit 1: RX-RZ layers, no entanglement
    # params shape: (reps, num_wires, 2)
    # ------------------------------------------------------------------
    if num == 1:
        def circ1(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i], wires)

        return circ1

    # ------------------------------------------------------------------
    # Circuit 2: RX-RZ + CNOT cascade (nearest-neighbor, fixed)
    # params shape: (reps, num_wires, 2)
    # ------------------------------------------------------------------
    elif num == 2:
        def circ2(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i], wires)
                CNOT_cascade(wires, ascending=True)

        return circ2

    # ------------------------------------------------------------------
    # Circuit 3: RX-RZ + parametrized CZ chain
    # params shape: (reps, num_wires, 2) for rotations + (reps, num_wires-1) for CZ angles
    # Packed as params shape: (reps, 2*num_wires + (num_wires-1))
    # ------------------------------------------------------------------
    elif num == 3:
        def circ3(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]  # params: (reps, num_wires, 3)
            # params[:, :, :2] = RX/RZ angles, params[:, :num_wires-1, 2] = CP angles
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                parametrized_CZ_chain(params[i, :num_wires - 1, 2], wires)

        return circ3

    # ------------------------------------------------------------------
    # Circuit 4: RX-RZ + parametrized CX (H-CP-H) chain
    # params shape: (reps, num_wires, 3)  — same layout as circuit 3
    # ------------------------------------------------------------------
    elif num == 4:
        def circ4(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                parametrized_CX_chain(params[i, :num_wires - 1, 2], wires)

        return circ4

    # ------------------------------------------------------------------
    # Circuit 5: RX-RZ, all-to-all parametrized CZ, RX-RZ
    # params shape: (reps, num_wires, 4 + num_wires - 1)
    # Layout per rep: [RX-RZ block] [all-to-all CZ] [RX-RZ block]
    # Simplified packing: params shape (reps, 4*num_wires + num_wires*(num_wires-1))
    # Here we use explicit named slices for clarity.
    # ------------------------------------------------------------------
    elif num == 5:
        def circ5(params, wires=None):
            """
            params shape: (reps, num_wires, 2 + (num_wires - 1) + 2)
            params[:, :, 0:2]                      = first RX-RZ block
            params[:, i, 2:2+(num_wires-1)]        = CP angles for qubit i to all others
            params[:, :, 2+(num_wires-1):]         = second RX-RZ block (2 angles)
            """
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            n_cp = num_wires - 1  # angles per qubit (skip self)
            for i in range(reps):
                # First RX-RZ
                rx_rz_layer(params[i, :, :2], wires)
                # All-to-all parametrized CZ
                for q in range(num_wires):
                    cp_idx = 0
                    for q2 in range(num_wires):
                        if q2 != q:
                            qp.ControlledPhaseShift(params[i, q, 2 + cp_idx], wires=[wires[q], wires[q2]])
                            cp_idx += 1
                # Second RX-RZ
                rx_rz_layer(params[i, :, 2 + n_cp:], wires)

        return circ5

    # ------------------------------------------------------------------
    # Circuit 6: RX-RZ, all-to-all parametrized CX (H-CP-H), RX-RZ
    # Same shape as circuit 5
    # ------------------------------------------------------------------
    elif num == 6:
        def circ6(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            n_cp = num_wires - 1
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                for q in range(num_wires):
                    cp_idx = 0
                    for q2 in range(num_wires):
                        if q2 != q:
                            qp.Hadamard(wires=wires[q])
                            qp.ControlledPhaseShift(params[i, q, 2 + cp_idx], wires=[wires[q], wires[q2]])
                            qp.Hadamard(wires=wires[q])
                            cp_idx += 1
                rx_rz_layer(params[i, :, 2 + n_cp:], wires)

        return circ6

    # ------------------------------------------------------------------
    # Circuit 7: RX-RZ, staggered parametrized CZ (even pairs then odd pairs), RX-RZ
    # params shape: (reps, num_wires, 2) + stagger CP angles
    # Packed: (reps, 4*num_wires + num_wires//2 + (num_wires-1)//2)
    # Simplified: params shape (reps, num_wires, 5)
    #   params[:, :, 0:2]              = first RX-RZ
    #   params[:, :num_wires//2, 5]    = even-pair CP angles
    #   params[:, :, 2:4]              = second RX-RZ
    #   params[:, :(num_wires-1)//2, 5]= odd-pair CP angles   (in separate tensor below)
    # For simplicity we split into separate param tensors matching the Qiskit structure.
    # ------------------------------------------------------------------
    elif num == 7:
        def circ7(params, wires=None):
            """
            params : (reps, num_wires, 6)
            """
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i,:,:2], wires)
                for q_idx, q in enumerate(range(0, num_wires - 1, 2)):
                    qp.ControlledPhaseShift(params[i, q_idx, 4], wires=[wires[q], wires[q + 1]])
                rx_rz_layer(params[i,:,2:4], wires)
                for q_idx, q in enumerate(range(1, num_wires - 1, 2)):
                    qp.ControlledPhaseShift(params[i, q_idx, 5], wires=[wires[q], wires[q + 1]])

        return circ7

    # ------------------------------------------------------------------
    # Circuit 8: same as 7 but with parametrized CX (H-CP-H)
    # ------------------------------------------------------------------
    elif num == 8:
        def circ8(params, wires=None):
            """
            params : (reps, num_wires, 6)
            """
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i,:,:2], wires)
                for q_idx, q in enumerate(range(0, num_wires - 1, 2)):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q_idx, 4], wires=[wires[q], wires[q + 1]])
                    qp.Hadamard(wires=wires[q])
                rx_rz_layer(params[i,:,2:4], wires)
                for q_idx, q in enumerate(range(1, num_wires - 1, 2)):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q_idx, 5], wires=[wires[q], wires[q + 1]])
                    qp.Hadamard(wires=wires[q])

        return circ8

    # ------------------------------------------------------------------
    # Circuit 9: H layer, CZ chain (fixed), RX layer
    # params shape: (reps, num_wires, 1) for RX angles
    # ------------------------------------------------------------------
    elif num == 9:
        def circ9(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for w in wires:
                    qp.Hadamard(wires=w)
                for q in range(num_wires - 1):
                    qp.CZ(wires=[wires[q], wires[q + 1]])
                for q in range(num_wires):
                    qp.RX(params[i, q, 0], wires=wires[q])

        return circ9

    # ------------------------------------------------------------------
    # Circuit 10: CZ chain with wrap-around + RY layer
    # params shape: (reps, num_wires, 1) for RY angles
    # ------------------------------------------------------------------
    elif num == 10:
        def circ10(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires - 1):
                    qp.CZ(wires=[wires[q], wires[q + 1]])
                qp.CZ(wires=[wires[num_wires - 1], wires[0]])
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])

        return circ10

    # ------------------------------------------------------------------
    # Circuit 11: RY-RZ, even CNOT, inner RY-RZ, odd CNOT
    # params shape: (reps, num_wires, 2) for first block
    #               (reps, num_wires-2, 2) for inner block
    # ------------------------------------------------------------------
    elif num == 11:
        def circ11(params, wires=None):
            """
            params shape: (reps, num_wires, 4)
              [:, :, 0:2] = outer RY+RZ on all wires
              [:, :, 2:4] = inner RY+RZ on inner wires (1..num_wires-2), independent slots
            """
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])
                    qp.RZ(params[i, q, 1], wires=wires[q])
                for q in range(0, num_wires - 1, 2):
                    qp.CNOT(wires=[wires[q], wires[q + 1]])
                for q_idx, q in enumerate(range(1, num_wires - 1)):
                    qp.RY(params[i, q_idx, 2], wires=wires[q])
                    qp.RZ(params[i, q_idx, 3], wires=wires[q])
                for q in range(1, num_wires - 1, 2):
                    qp.CNOT(wires=[wires[q], wires[q + 1]])

        return circ11

    # ------------------------------------------------------------------
    # Circuit 12: same as 11 but with CZ instead of CNOT
    # ------------------------------------------------------------------
    elif num == 12:
        def circ12(params, wires=None):
            """
            params shape: (reps, num_wires, 4), same layout as circuit 11
            (dims 2:4 independent for the inner layer).
            """
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])
                    qp.RZ(params[i, q, 1], wires=wires[q])
                for q in range(0, num_wires - 1, 2):
                    qp.CZ(wires=[wires[q], wires[q + 1]])
                for q_idx, q in enumerate(range(1, num_wires - 1)):
                    qp.RY(params[i, q_idx, 2], wires=wires[q])
                    qp.RZ(params[i, q_idx, 3], wires=wires[q])
                for q in range(1, num_wires - 1, 2):
                    qp.CZ(wires=[wires[q], wires[q + 1]])

        return circ12

    # ------------------------------------------------------------------
    # Circuit 13: RY, circular CP forward, RY, circular CP backward
    # params shape: (reps, num_wires, 4)
    #   [:, :, 0] = first RY, [:, :, 1] = CP forward
    #   [:, :, 2] = second RY, [:, :, 3] = CP backward
    # ------------------------------------------------------------------
    elif num == 13:
        def circ13(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])
                for q in range(num_wires):
                    qp.ControlledPhaseShift(params[i, q, 1], wires=[wires[q], wires[(q + 1) % num_wires]])
                for q in range(num_wires):
                    qp.RY(params[i, q, 2], wires=wires[q])
                for q in range(num_wires):
                    qp.ControlledPhaseShift(params[i, q, 3], wires=[wires[q], wires[(q - 1) % num_wires]])

        return circ13

    # ------------------------------------------------------------------
    # Circuit 14: same as 13 but with parametrized CX (H-CP-H) circular
    # params shape: (reps, num_wires, 4)
    # ------------------------------------------------------------------
    elif num == 14:
        def circ14(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])
                for q in range(num_wires):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q, 1], wires=[wires[q], wires[(q + 1) % num_wires]])
                    qp.Hadamard(wires=wires[q])
                for q in range(num_wires):
                    qp.RY(params[i, q, 2], wires=wires[q])
                for q in range(num_wires):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q, 3], wires=[wires[(q - 1) % num_wires], wires[q]])
                    qp.Hadamard(wires=wires[q])

        return circ14

    # ------------------------------------------------------------------
    # Circuit 15: RY, circular CNOT forward, RY, circular CNOT backward
    # params shape: (reps, num_wires, 2) 
    # ------------------------------------------------------------------
    elif num == 15:
        def circ15(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                for q in range(num_wires):
                    qp.RY(params[i, q, 0], wires=wires[q])
                for q in range(num_wires):
                    qp.CNOT(wires=[wires[q], wires[(q + 1) % num_wires]])
                for q in range(num_wires):
                    qp.RY(params[i, q, 1], wires=wires[q])
                for q in range(num_wires):
                    qp.CNOT(wires=[wires[(q - 1) % num_wires], wires[q]])

        return circ15

    # ------------------------------------------------------------------
    # Circuit 16: RX-RZ + staggered parametrized CZ sharing the same angle set
    # (even then odd pairs reuse the same param array — matches Qiskit bug/feature)
    # params shape: (reps, num_wires, 3)
    #   [:, :, :2] = RX-RZ, [:, :num_wires-1, 2] = CP angles (shared even/odd)
    # ------------------------------------------------------------------
    elif num == 16:
        def circ16(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                for q in range(0, num_wires - 1, 2):
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[q + 1]])
                for q in range(1, num_wires - 1, 2):
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[q + 1]])

        return circ16

    # ------------------------------------------------------------------
    # Circuit 17: same as 16 but with parametrized CX (H-CP-H)
    # ------------------------------------------------------------------
    elif num == 17:
        def circ17(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                for q in range(0, num_wires - 1, 2):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[q + 1]])
                    qp.Hadamard(wires=wires[q])
                for q in range(1, num_wires - 1, 2):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[q + 1]])
                    qp.Hadamard(wires=wires[q])

        return circ17

    # ------------------------------------------------------------------
    # Circuit 18: RX-RZ + circular parametrized CZ (wraps around)
    # params shape: (reps, num_wires, 3)
    #   [:, :, :2] = RX-RZ, [:, :, 2] = CP angles (including wrap-around edge)
    # ------------------------------------------------------------------
    elif num == 18:
        def circ18(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                for q in range(num_wires):
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[(q + 1) % num_wires]])

        return circ18

    # ------------------------------------------------------------------
    # Circuit 19: RX-RZ + circular parametrized CX (H-CP-H, wraps around)
    # params shape: (reps, num_wires, 3)  — same layout as circuit 18
    # ------------------------------------------------------------------
    elif num == 19:
        def circ19(params, wires=None):
            reps, num_wires = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_wires))
            for i in range(reps):
                rx_rz_layer(params[i, :, :2], wires)
                for q in range(num_wires):
                    qp.Hadamard(wires=wires[q])
                    qp.ControlledPhaseShift(params[i, q, 2], wires=[wires[q], wires[(q + 1) % num_wires]])
                    qp.Hadamard(wires=wires[q])

        return circ19


    # ------------------------------------------------------------------
    # Circuit 30: Strongly entangling layers (PennyLane built-in)
    # params shape: (num_layers, num_wires, 3)
    # ------------------------------------------------------------------
    elif num == 30:
        return qp.StronglyEntanglingLayers

    # ------------------------------------------------------------------
    # Circuit 31: Perfect SU(4) layers
    # params shape: (num_layers, num_wires//2, 3, 3)
    # ------------------------------------------------------------------

    elif num == 31:
        def perfect_SU4(params, wires = None):
            """ 
            params : tenor of shape (num_layers, num_wires//2, 3, 3)
            """
            if wires is None:
                wires = list(range(params.shape[1]*2))
            num_wires = len(wires)
            num_layers, num_pairs = params.shape[0], params.shape[1]
            wires_parity = 1 - (num_wires % 2)
            #print(f"num_layers: {num_layers}, num_pairs: {num_pairs}, wires_parity: {wires_parity}")
            for layer in range(num_layers):
                layer_pairs = num_pairs - layer % 2 * wires_parity
                layer_wires = wires[layer % 2: layer % 2 + layer_pairs * 2]
                #print(f"Layer {layer}: applying SU4 on wires {layer_wires}, pairs: {layer_pairs}, params shape: {params[layer].shape}")
                for i in range(layer_pairs):
                    #print(f"              Applying full_SU2 on wires {layer_wires[2 * i]} and {layer_wires[2 * i + 1]} with params {params[layer, i, :2]}")
                    full_SU2(params[layer, i, :2], wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])

                for i in range(layer_pairs):
                    qp.CNOT(wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])
                    qp.RZ(params[layer, i, 2, 0], wires=layer_wires[2 * i])
                    qp.RY(params[layer, i, 2, 1], wires=layer_wires[2 * i + 1])
                    qp.CNOT(wires=[layer_wires[2 * i + 1], layer_wires[2 * i]])
                    qp.RY(params[layer, i, 2, 2], wires=layer_wires[2*i + 1])
                    qp.CNOT(wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])

                # With repetitive layers this is redondant with the next layer.
                #for i in range(num_wires//2):
                #    full_SU2(params[layer, i, 3:5], wires=[wires[2 * i], wires[2 * i + 1]])
        return perfect_SU4
    # ------------------------------------------------------------------
    # Circuit 32: Two rotations + staggered CNOT (brickwall)
    # params shape: (num_layers, num_wires, 2)
    # ------------------------------------------------------------------

    elif num == 32:
        def two_rotations_brickwall(params, wires = None):
            """ 
            params : tenor of shape (num_layers, num_wires, 2)
            """
            num_layers, num_wires = params.shape[0], params.shape[1]

            if wires is None:
                wires = list(range(num_wires))
            num_wires = len(wires)
            num_pairs = num_wires // 2
            wires_parity = 1 - (num_wires % 2)


            for layer in range(num_layers):
                layer_pairs = num_pairs - layer % 2 * wires_parity
                layer_wires = wires[layer % 2: layer % 2 + layer_pairs * 2]
                two_rotations(params[layer, : layer_pairs * 2], wires=layer_wires)
                for i in range(layer_pairs):
                    qp.CNOT(wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])

        return two_rotations_brickwall

    # ------------------------------------------------------------------
    # Circuit 33: KAK1 exact-Haar block (2 qubits only)
    # params shape: (reps, 1, 15)
    # Tucci's KAK1: U = (A1 x A0) exp(i(k1 XX + k2 YY + k3 ZZ)) (B1 x B0),
    # fed raw Uniform(0, 2*pi) parameters (as sample_unitaries already
    # generates for every circuit here) and reparametrized via
    # haar_reparam so the resulting 2-qubit unitary is exactly
    # Haar-distributed on SU(4). See haar_reparam.py for the derivation
    # and validation.
    # ------------------------------------------------------------------
    elif num == 33:
        def kak1_haar(params, wires=None):
            """params : tensor of shape (reps, 1, 15) [+ optional trailing batch dim]"""
            reps = params.shape[0]
            if wires is None:
                wires = [0, 1]
            for layer in range(reps):
                kak1_haar_block(params[layer, 0], wires=[wires[0], wires[1]])

        return kak1_haar

    # ------------------------------------------------------------------
    # Circuit 34: KAK1 exact-Haar block, brickwork (N qubits)
    # params shape: (reps, num_wires // 2, 15)
    # Same 15-parameter Haar-exact 2-qubit block as circuit 33, applied to
    # adjacent-pair "bricks" that alternate offset by one wire each layer
    # (same brick pattern as circuit 32). Reduces to circuit 33 exactly
    # when num_wires == 2.
    # ------------------------------------------------------------------
    elif num == 34:
        def kak1_haar_brickwall(params, wires=None):
            """params : tensor of shape (reps, num_wires // 2, 15) [+ optional trailing batch dim]"""
            num_layers, num_pairs = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_pairs * 2))
            num_wires = len(wires)
            wires_parity = 1 - (num_wires % 2)

            for layer in range(num_layers):
                layer_pairs = num_pairs - layer % 2 * wires_parity
                layer_wires = wires[layer % 2: layer % 2 + layer_pairs * 2]
                for i in range(layer_pairs):
                    kak1_haar_block(params[layer, i], wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])

        return kak1_haar_brickwall

    # ------------------------------------------------------------------
    # Circuit 35: KAK1 ablation -- same block as 33, no reparametrization
    # params shape: (reps, 1, 15)
    # Identical gate structure to circuit 33 (4 local SU(2) + 3-CNOT core),
    # but the raw Uniform(0, 2*pi) parameters are used directly as gate
    # angles instead of being pushed through haar_reparam. Compare F^(t)
    # against circuit 33 to quantify what the reparametrization buys you.
    # ------------------------------------------------------------------
    elif num == 35:
        def kak1_uniform(params, wires=None):
            """params : tensor of shape (reps, 1, 15) [+ optional trailing batch dim]"""
            reps = params.shape[0]
            if wires is None:
                wires = [0, 1]
            for layer in range(reps):
                kak1_block_naive(params[layer, 0], wires=[wires[0], wires[1]])

        return kak1_uniform

    # ------------------------------------------------------------------
    # Circuit 36: KAK1 ablation, brickwork -- same as 34, no reparametrization
    # params shape: (reps, num_wires // 2, 15)
    # Brickwork counterpart of circuit 35, exactly as 34 is to 33.
    # ------------------------------------------------------------------
    elif num == 36:
        def kak1_uniform_brickwall(params, wires=None):
            """params : tensor of shape (reps, num_wires // 2, 15) [+ optional trailing batch dim]"""
            num_layers, num_pairs = params.shape[0], params.shape[1]
            if wires is None:
                wires = list(range(num_pairs * 2))
            num_wires = len(wires)
            wires_parity = 1 - (num_wires % 2)

            for layer in range(num_layers):
                layer_pairs = num_pairs - layer % 2 * wires_parity
                layer_wires = wires[layer % 2: layer % 2 + layer_pairs * 2]
                for i in range(layer_pairs):
                    kak1_block_naive(params[layer, i], wires=[layer_wires[2 * i], layer_wires[2 * i + 1]])

        return kak1_uniform_brickwall

    else:
        raise ValueError(f"Circuit number {num} is not defined.")
    

def weight_tensor_shape(num, num_wires, reps = 1):
    # Return the needed shape of the weight tensor for a given circuit number
    if num == 1:
        return (reps, num_wires, 2)
    elif num == 2:
        return (reps, num_wires, 2)
    elif num == 3:
        return (reps, num_wires, 3)
    elif num == 4:  
        return (reps, num_wires, 3)
    elif num == 5:
        return (reps, num_wires, 4 + num_wires - 1)
    elif num == 6:
        return (reps, num_wires, 4 + num_wires - 1)
    elif num == 7:
        return (reps, num_wires, 6)
    elif num == 8:
        return (reps, num_wires, 6)
    elif num == 9:
        return (reps, num_wires, 1)
    elif num == 10:
        return (reps, num_wires, 1)
    elif num == 11:
        return (reps, num_wires, 4)
    elif num == 12:
        return (reps, num_wires, 4)
    elif num == 13:
        return (reps, num_wires, 4)
    elif num == 14:
        return (reps, num_wires, 4)
    elif num == 15:
        return (reps, num_wires, 2)
    elif num == 16:
        return (reps, num_wires, 3)
    elif num == 17:
        return (reps, num_wires, 3)
    elif num == 18:
        return (reps, num_wires, 3)
    elif num == 19:
        return (reps, num_wires, 3)
    elif num == 30:
        return (reps, num_wires, 3)
    elif num == 31:
        return (reps, num_wires//2, 3, 3)
    elif num == 32:
        return (reps, num_wires, 2)
    elif num == 33:
        return (reps, 1, 15)
    elif num == 34:
        return (reps, num_wires // 2, 15)
    elif num == 35:
        return (reps, 1, 15)
    elif num == 36:
        return (reps, num_wires // 2, 15)
    else:
        raise ValueError(f"Circuit number {num} is not defined.")


def n_trainable(num, num_wires, reps=1):
    """Number of weight-tensor entries that circuit_set(num) actually reads.

    weight_tensor_shape allocates a rectangular tensor, but several circuits
    don't read all of it. Rather than hand-count the
    used entries per circuit (error-prone and easy to let drift out of sync
    with circuit_set), this measures it: differentiate a random linear
    functional of the full output distribution w.r.t. every weight entry and
    count the nonzero gradients. A structurally-unused entry gets an exact
    zero gradient; a coincidental zero from a used entry has probability zero
    under random projection weights and random input angles.
    """
    shape = weight_tensor_shape(num, num_wires, reps)
    generator = torch.Generator().manual_seed(0)
    weights = 2 * torch.pi * torch.rand(shape, dtype=torch.float64, generator=generator)
    weights.requires_grad_(True)
    dev = qp.device("default.qubit", wires=num_wires)

    @qp.qnode(dev, interface="torch", diff_method="backprop")
    def probe(weights):
        circuit_set(num=num)(weights, wires=list(range(num_wires)))
        # qp.state() (not qp.probs()) so phase-only gates (e.g. a trailing RZ)
        # are visible too — probabilities in the computational basis are blind
        # to them, which would falsely mark their parameters as unused.
        return qp.state()

    state = probe(weights)
    real_projection = torch.rand(2 ** num_wires, dtype=torch.float64, generator=generator)
    imag_projection = torch.rand(2 ** num_wires, dtype=torch.float64, generator=generator)
    loss = state.real @ real_projection + state.imag @ imag_projection
    loss.backward()
    return int((weights.grad.abs() > 1e-9).sum().item())