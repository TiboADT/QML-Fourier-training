import torch
import pennylane as qp


degree = 1  # degree of the target function
scaling = 1  # scaling of the data
coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
coeff0 = 0.1  # coefficient of zero frequency


def target_function(x):
    x_c = x.to(torch.complex64)
    """Generate a truncated Fourier series, where the data gets re-scaled."""
    res = torch.full_like(x_c, fill_value=coeff0, dtype=torch.complex64)
    for idx, coeff in enumerate(coeffs):
        k = idx + 1
        coeff_t = torch.as_tensor(coeff, dtype=torch.complex64, device=x.device)
        exponent = 1j * k * x_c
        res = res + coeff_t * torch.exp(exponent) + torch.conj(coeff_t) * torch.exp(-exponent)

    return torch.real(res)


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

dev = qp.device('default.qubit', wires=2)

def S(x):
    """Data-encoding circuit block."""
    qp.RX(x, wires=0)


def W(theta):
    """Trainable circuit block."""
    qp.Rot(theta[0], theta[1], theta[2], wires=0)


@qp.qnode(dev, interface="torch")
def serial_quantum_model(weights, x):

    for theta in weights[:-1]:
        W(theta)
        S(x)

    # (L+1)'th unitary
    W(weights[-1])

    return qp.expval(qp.PauliZ(wires=0))

weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
print( weights)

r = 1  # number of times the encoding gets repeated (here equal to the number of layers)
weights = (
    2 * torch.pi * torch.rand((r + 1, 3), requires_grad=True)
)  # some random initial weights

print(weights)

weights = weights.detach().clone().requires_grad_(True)

print(weights)

x = torch.linspace(-6, 6, 70, requires_grad=False)
y = target_function(x)


opt = torch.optim.Adam([weights], lr = 0.1)

steps = 200

cost = cost_model(serial_quantum_model)

def closure():
    opt.zero_grad()
    loss = cost(weights, x, y)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)
    print(f"Step {i+1}: weights = {weights.detach().numpy()}, cost = {cost(weights, x, y).item()}")