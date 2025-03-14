from qiskit import QuantumCircuit

from qlearning.circuit.base import MeasurementBlock


class FirstQubitMeasurement(MeasurementBlock):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'FirstQubitMeasurement')
        self._circuit = QuantumCircuit(num_qubits, 1)
        self._circuit.measure(0, 0)
