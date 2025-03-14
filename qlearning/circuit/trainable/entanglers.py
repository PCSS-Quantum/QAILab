from qlearning.circuit.base import EntanglingBlock


class CXEntangler(EntanglingBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'CXEntangler')
        for i in range(1, num_qubits):
            self._circuit.cx(i-1, i)

        self._circuit.cx(num_qubits-1, 0)
