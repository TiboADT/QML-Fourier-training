"""Architecture: one small object bundling a circuit_set number (or
generic_ansatz code) with whether it's generic, to simplify the interface to the rest of the code.
"""

from dataclasses import dataclass

import circuits
import generic_ansatz


@dataclass(frozen=True)
class Architecture:
    num: int
    generic: bool = False

    def build(self, params, wires=None):
        """Queue the gate sequence for this architecture."""
        if self.generic:
            generic_ansatz.build(self.num, params, wires=wires)
        else:
            circuits.circuit_set(num=self.num)(params, wires=wires)

    def weight_shape(self, n_qubits, reps=1):
        """The params tensor shape this architecture needs."""
        if self.generic:
            return generic_ansatz.weight_tensor_shape(self.num, n_qubits, reps)
        return circuits.weight_tensor_shape(self.num, n_qubits, reps)

    def n_trainable(self, n_qubits, reps=1):
        """How many of weight_shape's entries are actually read -- exact
        for a generic code (generic_ansatz.n_trainable has a closed form),
        a gradient probe for a legacy circuit (circuits.n_trainable)."""
        if self.generic:
            return generic_ansatz.n_trainable(self.num, n_qubits, reps)
        return circuits.n_trainable(self.num, n_qubits, reps)

    @property
    def label(self) -> str:
        return str(self.num) if self.generic else f"circuit {self.num}"


def get_architecture(num: int, generic: bool = False) -> Architecture:
    return Architecture(num=num, generic=generic)
