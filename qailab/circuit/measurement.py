"""Block implementing circuit measurement"""

from qiskit import QuantumCircuit

from qailab.circuit.base import NonGateBlock


class MeasurementBlock(NonGateBlock):
    """Blocks defining the output structure"""

    def __init__(self) -> None:
        super().__init__('MeasurementBlock')

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits, num_qubits, name=self.name)
        for i in range(num_qubits):
            circuit.measure(i, i)
        return circuit
