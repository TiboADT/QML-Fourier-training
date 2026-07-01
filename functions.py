import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt

import torch

from circuits import circuit_set, weight_tensor_shape

# ── Device setup ──────────────────────────────────────────────────────────────

def get_device(verbose: bool = False) -> torch.device:
    """
    Return the best available device, checked in this order:
        1. CUDA  (NVIDIA GPU)
        2. XPU   (Intel GPU, requires intel-extension-for-pytorch)
        3. CPU   (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(device)
            vram  = props.total_memory / 1024**3
            print(f"GPU found : {props.name}  (CUDA)")
            print(f"VRAM      : {vram:.1f} GB")
            print(f"CUDA      : {torch.version.cuda}")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        if verbose:
            props = torch.xpu.get_device_properties(device)
            vram  = props.total_memory / 1024**3
            print(f"GPU found : {props.name}  (Intel XPU)")
            print(f"VRAM      : {vram:.1f} GB")

    else:
        device = torch.device("cpu")
        if verbose:
            print("No GPU found — running on CPU (no speedup vs original script)")

    return device


def _get_vram_gb(device: torch.device) -> float:
    """Return total VRAM in GB for a CUDA or XPU device."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / 1024**3
    elif device.type == "xpu":
        return torch.xpu.get_device_properties(device).total_memory / 1024**3
    return 0.0

# ── Training and cost functions ───────────────────────────────────────────────

def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)
    return 0.5 * loss


def cost_model(model):
    def cost(weights, x, y):
        predictions = model(weights, x)
        return square_loss(y, predictions)
    return cost


def train(model, weights, x, target_y, max_steps=70, batch_size=50, display_step=10, display=True):
    weights = weights.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([weights], lr = 0.1)
    cost = cost_model(model)
    cst = torch.zeros((max_steps), dtype=torch.float32)
    with torch.no_grad():
        cst[0] = cost(weights, x, target_y)  # initial cost

    def closure():
        opt.zero_grad()
        loss = cost(weights, x_batch, y_batch)
        loss.backward()
        return loss
    
    for step in range(max_steps):
        # select batch of data using torch's random permutation
        perm = torch.randperm(len(x))
        x_batch = x[perm[:batch_size]]
        y_batch = target_y[perm[:batch_size]]

        # update the weights by one optimizer step

        opt.step(closure)

        # save, and possibly print, the current cost
        cst[step] = cost(weights, x, target_y).detach().item()
        if (step + 1) % display_step == 0 and display:
            print("                 Cost at step {0:3}: {1}".format(step + 1, cst[step]))
    return weights.detach(), cst


def build_model(circuit_num, n_qubits, layers, anzats_reps = 1, measuring_qubit = 0):
    """Build a model for the given circuit number, number of qubits, and repetitions."""
    
    circuit_to_call = circuit_set(num=circuit_num)

    weights_shape = weight_tensor_shape(circuit_num, n_qubits, anzats_reps)
    weights_shape = (layers,) + weights_shape
    weights = 2 * torch.pi * torch.rand(weights_shape, requires_grad=True)

    dev = qp.device("default.qubit", wires=n_qubits, shots=None)

    def S(x):
        """Data-encoding circuit block."""
        for w in range(n_qubits):
            qp.RX(x, wires=w)

    @qp.qnode(dev, interface="torch")
    def circuit(weights,x):
        (layers,trainable_block_layers,qubits) = weights.shape[:3]
        layers = layers - 1
        wires = list(range(n_qubits))
        circuit_to_call(weights[0], wires = wires)
        for l in range(layers):
            S(x*(l+1))
            circuit_to_call(weights[l+1], wires = wires)

        return qp.expval(qp.PauliZ(wires=measuring_qubit))

    return circuit, weights


def show_results(model,weights, x, target_y, cst, title="Results"):
    """Helper function to visualize the results."""
    predictions = model(weights, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title)

    # plot the target function and the model predictions
    ax1.plot(x, predictions.detach().numpy(), c="blue")
    ax1.scatter(x, target_y, facecolor="white", edgecolor="black")
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")

    # plot the cost in a logarithmic scale
    ax2.plot(cst.detach().numpy(), c="red")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cost")
    ax2.set_yscale("log")
    plt.show()

def function_to_learn(degree = 1, coeffs = None, coeff0 = 0.1):
    if coeffs is None:
        coeffs = np.random.random(size=degree) + 1j * np.random.random(size=degree)  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency
    coeffs = coeffs / np.sum(np.abs(coeffs))
    coeffs = coeffs * (1 - coeff0) / 2 

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = torch.full_like(x, fill_value=coeff0, dtype=torch.complex64)
        for idx, coeff in enumerate(coeffs):
            k = idx + 1
            coeff_t = torch.as_tensor(coeff, dtype=torch.complex64, device=x.device)
            exponent = 1j * k * x
            res = res + coeff_t * torch.exp(exponent) + torch.conj(coeff_t) * torch.exp(-exponent)

        return torch.real(res)
    
    return target_function

def coucou():
    print("et non en fait")