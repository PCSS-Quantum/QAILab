"""Different implementations of qubit entangling sections for variational circuits."""
from qlearning.circuit.base import EntanglingBlock


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

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'CXEntangler')

    def _build_circuit(self):
        for i in range(1, self.num_qubits):
            self._circuit.cx(i-1, i)

        self._circuit.cx(self.num_qubits-1, 0)
