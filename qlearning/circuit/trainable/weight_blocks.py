from qlearning.circuit.base import WeightBlock
from qiskit import QuantumCircuit


class RyWeight(WeightBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'RyWeight')
        self._circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self._circuit.ry(self.weights_pv[i], i)
