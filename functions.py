import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt

import torch

from circuits import circuit_set, weight_tensor_shape

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
    """`model` (from build_model) counts expectation-value estimates as it's
    called -- see build_model's counted_circuit. That count is exact for
    training done here specifically because this device+interface
    combination resolves to backprop differentiation: .backward() reuses
    the one forward pass's autodiff graph rather than triggering further
    device executions, so every expectation value genuinely estimated
    corresponds to exactly one `model(weights, x)` call below (never a
    hidden multiple, the way parameter-shift gradients would need)."""
    weights = weights.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([weights], lr = 0.1)
    cost = cost_model(model)
    # cst[0] is the pre-training cost, cst[1..max_steps] the cost after each
    # optimizer step — keeping both means neither overwrites the other.
    cst = torch.zeros((max_steps + 1), dtype=torch.float32)
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
        cst[step + 1] = cost(weights, x, target_y).detach().item()
        if (step + 1) % display_step == 0 and display:
            print("                 Cost at step {0:3}: {1}".format(step + 1, cst[step + 1]))
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
            S(x)
            circuit_to_call(weights[l+1], wires = wires)

        return qp.expval(qp.PauliZ(wires=measuring_qubit))

    # Every call above is one simulated device execution, but (thanks to
    # PennyLane's parameter broadcasting) it covers x.shape[0] independent
    # expectation-value estimates at once -- on real hardware that would be
    # x.shape[0] separate circuit runs, not one. A step count doesn't
    # reflect this: it's the same whether batch_size is 10 or 1000, and
    # doesn't vary with n_qubits/circuit depth either. This does (no
    # additional device executions are hiding inside .backward()).
    def counted_circuit(weights, x):
        counted_circuit.n_expvals += x.shape[0]
        return circuit(weights, x)
    counted_circuit.n_expvals = 0

    return counted_circuit, weights


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