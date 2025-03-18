"""Build parameterized QuantumCircuits from lists of block types."""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qlearning.circuit.base import LayerBlock, WeightBlock, EncodingBlock, MeasurementBlock


def build_circuit(
    input_size: int,
    circuit_layers: list[type[LayerBlock]],
    encoding_style: type[EncodingBlock],
    measurement_style: type[MeasurementBlock],
    # * **kwargs  WIP
) -> tuple[QuantumCircuit, list[ParameterVector], ParameterVector]:
    """
    Builds a parameterized QuantumCircuit.

    Args:
        input_size (int): How many qubits to use for the circuit.
        circuit_layers (list[type[LayerBlock]]): How to encode weights, perform entanglement, do dropout etc.
        encoding_style (type[EncodingBlock]): How to encode the input vector.
        measurement_style (type[MeasurementBlock]): What to measure: first qubit, all qubits etc.

    Returns:
        tuple[QuantumCircuit, list[ParameterVector], ParameterVector]:
                Built circuit, list of parameter vectors for weight layers, parameter vector for input (x)
    """
    circuit = QuantumCircuit(input_size)
    weight_vectors = []

    encoding_block = encoding_style(input_size)
    encoding_block.add_to_circuit(circuit)

    for layer in circuit_layers:
        block = layer(input_size)
        block.add_to_circuit(circuit)

        if isinstance(block, WeightBlock):
            weight_vectors.append(block.weights)

    measurement_block = measurement_style(input_size)
    measurement_block.add_to_circuit(circuit)

    return circuit, weight_vectors, encoding_block.input
