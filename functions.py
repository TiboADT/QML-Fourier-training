import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt

import torch

from circuits import circuit_set, weight_tensor_shape


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


def train(model, weights, x, target_y, max_steps=70, batch_size=25, display_step=10, display=True):
    weights = weights.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([weights], lr = 0.1)
    cost = cost_model(model)
    cst = torch.zeros((max_steps), dtype=torch.float32)
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
        c = cost(weights, x, target_y)
        cst[step] = c
        if (step + 1) % display_step == 0 and display:
            print("Cost at step {0:3}: {1}".format(step + 1, c))
    return weights, cst


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
        (layers,trainable_block_layers,qubits,_) = weights.shape
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


def coucou():
    print("et non en fait")