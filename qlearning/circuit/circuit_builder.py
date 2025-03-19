"""Build parameterized QuantumCircuits from lists of blocks."""
from typing import Sequence, Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.quantumcircuit import QubitSpecifier

from qlearning.circuit.measurement import MeasurementBlock
from qlearning.circuit.base import CircuitBlock, EncodingBlock


def _dedup(var_list: list[Any]) -> list[Any]:
    return list(dict.fromkeys(var_list))


def build_circuit(
    circuit_width: int,
    input_encoding_blocks: list[EncodingBlock] | None = None,
    layer_blocks: list[CircuitBlock] | None = None,
    measure_qubits: Sequence[QubitSpecifier] | None = None,
    # * **kwargs  WIP
) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector]]:
    """
    Builds a parameterized QuantumCircuit.

    Args:
        circuit_width (int): Number of qubits used for the circuit.
        input_encoding_blocks (list[EncodingBlock] | None, optional): Blocks encoding the input vector. Defaults to None.
        layer_blocks (list[CircuitBlock] | None, optional): Blocks encoding the weights, doing entanglement etc. Defaults to None.
        measure_qubits (Sequence[QubitSpecifier] | None, optional):
        Which qubits to measure. If None, measure all, except auxiliary. Defaults to None.

    Returns:
        tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector]]:
        Built circuit, input encoding parameters, weight encoding parameters.
    """
    circuit = QuantumCircuit(circuit_width)

    if input_encoding_blocks is None:
        input_encoding_blocks = []

    if layer_blocks is None:
        layer_blocks = []

    qargs = list(range(circuit_width))

    input_vectors = []
    for block in input_encoding_blocks:
        block.add_to_circuit(circuit, qargs)
        input_vectors.append(block.parameter_vector)

    weight_vectors = []
    for block in layer_blocks:
        block.add_to_circuit(circuit, qargs)

        if isinstance(block, EncodingBlock):
            weight_vectors.append(block.parameter_vector)

    measurement_block = MeasurementBlock()
    measurement_block.add_to_circuit(circuit, measure_qubits if measure_qubits is not None else qargs)

    return circuit, _dedup(input_vectors), _dedup(weight_vectors)  # Return only unique parameter vectors
