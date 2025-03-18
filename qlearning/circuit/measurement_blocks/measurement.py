"""Different measurement styles for variational circuits"""
from qiskit import QuantumCircuit

from qlearning.circuit.base import MeasurementBlock


class FirstQubitMeasurement(MeasurementBlock):
    """Measure only qubit 0 -> output shape = (2,)"""

    def __init__(self) -> None:
        super().__init__('FirstQubitMeasurement')

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits, 1, name=self.name)  # add classical bit
        circuit.measure(0, 0)
        return circuit


class AllQubitMeasurement(MeasurementBlock):
    """Measure all qubits -> output shape = 2^n"""

    def __init__(self) -> None:
        super().__init__('AllQubitMeasurement')

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits, num_qubits, name=self.name)  # add classical bits
        circuit.measure_all(add_bits=False)
        return circuit
