"""Build parameterized QuantumCircuits from lists of blocks."""
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import QubitSpecifier

from qailab.circuit.measurement import MeasurementBlock
from qailab.circuit.base import CircuitBlock


def build_circuit(
    circuit_width: int,
    blocks: list[CircuitBlock] | None = None,
    measure_qubits: Sequence[QubitSpecifier] | None = None,
    # * **kwargs  WIP
) -> QuantumCircuit:
    """
    Builds a parameterized QuantumCircuit.

    Args:
        circuit_width (int): Number of qubits used for the circuit (not including auxiliary qubits).
        blocks (list[CircuitBlock] | None, optional): Blocks making up the circuit. Defaults to None.
        measure_qubits (Sequence[QubitSpecifier] | None, optional):
        Which qubits to measure. If None, measure all, except auxiliary. Defaults to None.

    Returns:
        QuantumCircuit: Built circuit
    """
    circuit = QuantumCircuit(circuit_width)

    if blocks is None:
        blocks = []

    qargs = list(range(circuit_width))

    for block in blocks:
        block.add_to_circuit(circuit, qargs)

    measurement_block = MeasurementBlock()
    measurement_block.add_to_circuit(circuit, measure_qubits if measure_qubits is not None else qargs)

    return circuit
