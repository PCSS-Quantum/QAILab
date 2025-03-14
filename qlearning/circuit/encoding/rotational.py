from qlearning.circuit.base import EncodingBlock


class RxEncoder(EncodingBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'ExpXEncoder')
        for i in range(num_qubits):
            self._circuit.rx(self.x_pv[i], i)
