"""Different implementations of weight(trainable) encoding blocks for variational circuits."""
from qlearning.circuit.base import WeightBlock


class RxWeight(WeightBlock):
    """Encode each weight with an RX gate (weight is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RxWeight')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.rx(self.weights[i], i)


class RyWeight(WeightBlock):
    """Encode each weight with an RY gate (weight is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RyWeight')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.ry(self.weights[i], i)


class RzWeight(WeightBlock):
    """Encode each weight with an RZ gate (weight is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RzWeight')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.ry(self.weights[i], i)
