"""R-gate implementations of input vector encoding blocks for variational circuits."""
from qlearning.circuit.base import EncodingBlock


class RxEncoder(EncodingBlock):
    """Encode each part of x with an RX gate (x_i is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RxEncoder')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.rx(self.x_pv[i], i)


class RyEncoder(EncodingBlock):
    """Encode each part of x with an RY gate (x_i is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RyEncoder')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.ry(self.x_pv[i], i)


class RzEncoder(EncodingBlock):
    """Encode each part of x with an RZ gate (x_i is theta)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RzEncoder')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.rz(self.x_pv[i], i)
