from qlearning.circuit.base import WeightBlock


class RyWeight(WeightBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RyWeight')

    def _build_circuit(self):
        for i in range(self.num_qubits):
            self._circuit.ry(self.weights_pv[i], i)
