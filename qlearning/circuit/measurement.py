"""Block implementing circuit measurement"""
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import QubitSpecifier

from qlearning.circuit.base import CircuitBlock


class MeasurementBlock(CircuitBlock):
    """Blocks defining the output structure"""

    def __init__(self) -> None:
        super().__init__('MeasurementBlock')

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits, num_qubits, name=self.name)
        for i in range(num_qubits):
            circuit.measure(i, i)
        return circuit

    def add_to_circuit(self, circuit: QuantumCircuit, qargs: Sequence[QubitSpecifier] | None = None) -> None:
        if qargs is None:
            qargs = list(range(circuit.num_qubits))
        else:
            qargs = list(qargs)

        c = self._build_circuit(len(qargs))

        circuit.compose(c, qargs, inplace=True)
