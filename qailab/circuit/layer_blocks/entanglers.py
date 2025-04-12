"""Different implementations of qubit entangling sections for variational circuits."""
from qiskit import QuantumCircuit
from qailab.circuit.base import EntanglingBlock


class CXEntangler(EntanglingBlock):
    """
                        ┌───┐
    q_0: ──■────────────┤ X ├
         ┌─┴─┐          └─┬─┘
    q_1: ┤ X ├──■─────────┼──
         └───┘┌─┴─┐       │
    q_2: ─────┤ X ├──■────┼──
    ..        └───┘┌─┴─┐  │
    q_n: ──────────┤ X ├──■──
                   └───┘
    """

    def __init__(self) -> None:
        super().__init__('CXEntangler')

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)
        for i in range(1, num_qubits):
            circuit.cx(i-1, i)

        circuit.cx(num_qubits-1, 0)
        return circuit
