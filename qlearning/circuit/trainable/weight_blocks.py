from qlearning.circuit.base import WeightBlock


class RyWeight(WeightBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RyWeight')
        for i in range(num_qubits):
            self._circuit.ry(self.weights_pv[i], i)
