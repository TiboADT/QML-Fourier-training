import pennylane as qp

def circuit(name : str = None, num : str = None):
    """This function will take either the name of a circuit or it number and return a function corresponding to that circuit.
    """
    circuits = {
        "SU2": 31,
        "Brickwall": 31,
    }
    if num is None:
        if name is None:
            raise ValueError("You must provide either a circuit name or a circuit number.")
        num = circuits[name]

def full_SU2(params, wires = None):
    """ 
    params : tenor of shape (num_layers, num_wires, 3)
    """
    if wires is None:
        wires = list(range(params.shape[1]))
    num_layers, num_wires, _ = params.shape
    for layer in range(num_layers):
        for wire in wires:
            qp.Rot(*params[layer, wire], wires=wire)

def two_rotations(params, wires = None):
    """ 
    params : tenor of shape (num_layers, num_wires, 2)
    """
    if wires is None:
        wires = list(range(params.shape[1]))
    num_layers, num_wires, _ = params.shape
    for layer in range(num_layers):
        for wire in wires:
            qp.Rot(params[layer, wire, 0], params[layer, wire, 1], 0.0, wires=wire)

def CNOT_cascade(wires , circulare = False, ascending = False):
    """ 
    wires : list of wires to apply the CNOT cascade on
    circulare : if True, apply a CNOT from the last wire to the first wire
    ascending : if True, apply the CNOTs in ascending order, else in descending order
    """
    if ascending:
        for i in range(len(wires) - 1):
            qp.CNOT(wires=[wires[i], wires[i + 1]])
        if circulare:
            qp.CNOT(wires=[wires[-1], wires[0]])
    else:
        for i in range(len(wires) - 1, 0, -1):
            qp.CNOT(wires=[wires[i], wires[i - 1]])
        if circulare:
            qp.CNOT(wires=[wires[0], wires[-1]])
        
def perfect_SU4(params, wires = None):
    """ 
    params : tenor of shape (num_layers, num_wires//2, 15)
    """
    if wires is None:
        wires = list(range(params.shape[1]))
    num_layers, num_wires, _ = params.shape
    for layer in range(num_layers):
        full_SU2(params[layer, :, :3], wires=wires[::2])
        full_SU2(params[layer, :, 3:6], wires=wires[1::2])

        for i in range(num_wires // 2):
            qp.CNOT(wires=[wires[2 * i], wires[2 * i + 1]])
            qp.RZ(params[layer, i, 6], wires=wires[2 * i])
            qp.RY(params[layer, i, 7], wires=wires[2 * i + 1])
            qp.CNOT(wires=[wires[2 * i + 1], wires[2 * i]])
            qp.RY(params[layer, i, 8], wires=wires[2*i + 1])
            qp.CNOT(wires=[wires[2 * i], wires[2 * i + 1]])


        full_SU2(params[layer, :, 9:12], wires=wires[::2])
        full_SU2(params[layer, :, 12:15], wires=wires[1::2])
