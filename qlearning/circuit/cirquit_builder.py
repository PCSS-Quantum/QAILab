from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qlearning.circuit.base import CircuitBlock, WeightBlock, EncodingBlock, MeasurementBlock
from qlearning.circuit.trainable.entanglers import CXEntangler
from qlearning.circuit.trainable.weight_blocks import RyWeight
from qlearning.circuit.encoding.rotational import RxEncoder

LAYER_NAME = Literal['CXE', 'RyW']
ENCODING_NAME = Literal['RxE']
MEASUREMENT_NAME = Literal['none']


def _layer_name_to_type(name: LAYER_NAME) -> type[CircuitBlock]:
    match name:
        case 'CXE':
            return CXEntangler
        case 'RyW':
            return RyWeight
        case _:
            raise ValueError("name not supported")


def _encoding_name_to_type(name: ENCODING_NAME) -> type[EncodingBlock]:
    match name:
        case 'RxE':
            return RxEncoder
        case _:
            raise ValueError("name not supported")


def _measurement_name_to_type(name: MEASUREMENT_NAME) -> type[MeasurementBlock]:
    match name:
        case _:
            raise ValueError("name not supported")


def build_circuit(
    input_size: int,
    circuit_layers: list[Literal['CXE', 'RyW']],
    encoding_style: Literal['RxE'],
    measurement_style: Literal['none'],
    **kwargs
) -> tuple[QuantumCircuit, list[ParameterVector], ParameterVector]:

    circuit = QuantumCircuit(input_size)
    def c_add(block): return circuit.append(block.qiskit_gate, list(range(input_size)))
    weight_vectors = []

    for layer in circuit_layers:
        block = _layer_name_to_type(layer)(input_size)
        c_add(block)

        if isinstance(block, WeightBlock):
            weight_vectors.append(block.weights_pv)

    encoding_block = _encoding_name_to_type(encoding_style)(input_size)
    c_add(encoding_block)

    # TODO: add measurement blocks
    # measurement_block = _measurement_name_to_type(measurement_style)(input_size)

    return circuit, weight_vectors, encoding_block.x_pv
