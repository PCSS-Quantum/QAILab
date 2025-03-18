"""Different measurement styles for variational circuits"""
from qiskit import QuantumCircuit

from qlearning.circuit.base import MeasurementBlock


class FirstQubitMeasurement(MeasurementBlock):
    """Measure only qubit 0 -> output shape = (2,)"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'FirstQubitMeasurement')

    def _build_circuit(self):
        self._circuit = QuantumCircuit(self.num_qubits, 1, name=self.name)  # add classical bit
        self._circuit.measure(0, 0)


class AllQubitMeasurement(MeasurementBlock):
    """Measure all qubits -> output shape = 2^n"""

    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits, 'AllQubitMeasurement')

    def _build_circuit(self):
        self._circuit.measure_all(inplace=True)
