"""Build parameterized QuantumCircuits from lists of block types."""
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info.states.statevector import Statevector

from qlearning.circuit.base import CircuitBlock, EncodingBlock, MeasurementBlock


def build_circuit(
    circuit_width: int,
    input_encoding_blocks: list[EncodingBlock],
    layer_blocks: list[CircuitBlock],
    measurement_block: MeasurementBlock,
    initial_state: Statevector | Sequence[complex] | str | int | None = None,
    # * **kwargs  WIP
) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector]]:
    """
    Builds a parameterized QuantumCircuit.

    Args:
        circuit_width (int): Number of qubits used for the circuit.
        input_encoding_blocks (list[EncodingBlock]): Blocks encoding the input vector.
        layer_blocks (list[CircuitBlock]): Blocks encoding the weights, doing entanglement etc.
        measurement_block (MeasurementBlock): What type of measurement to use.
        initial_state (Statevector | Sequence[complex] | str | int | None, optional): Initial qubit state. Defaults to None.

    Returns:
        tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector]]:
        Built circuit, input encoding parameters, weight encoding parameters.
    """
    circuit = QuantumCircuit(circuit_width)

    if initial_state is not None:
        circuit.prepare_state(initial_state)

    input_vectors = []
    for block in input_encoding_blocks:
        block.add_to_circuit(circuit)
        input_vectors.append(block.parameter_vector)

    weight_vectors = []
    for block in layer_blocks:
        block.add_to_circuit(circuit)

        if isinstance(block, EncodingBlock):
            weight_vectors.append(block.parameter_vector)

    measurement_block.add_to_circuit(circuit)

    return circuit, input_vectors, weight_vectors
