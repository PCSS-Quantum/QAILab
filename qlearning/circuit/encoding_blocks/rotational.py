from qlearning.circuit.base import EncodingBlock


class RxEncoder(EncodingBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'ExpXEncoder')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.rx(self.x_pv[i], i)
