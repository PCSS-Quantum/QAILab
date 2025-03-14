from qiskit import QuantumCircuit

from qlearning.circuit.base import EncodingBlock


class RxEncoder(EncodingBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'ExpXEncoder')
        self._circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self._circuit.rx(self.x_pv[i], i)
