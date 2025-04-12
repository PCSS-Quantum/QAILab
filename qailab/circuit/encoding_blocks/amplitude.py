"""Blocks encoding values as initial state for a quantum circuit"""
from typing import Literal
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.circuit.library import RawFeatureVector

from qailab.circuit.base import EncodingBlock, NonGateBlock


class TypedRawFeatureVector(RawFeatureVector):
    """Extension of RawFeatureVector that renames the parameters."""

    def __init__(self, feature_dimension: int | None, block_type: Literal['input'] | Literal['weight'] = 'input') -> None:
        super().__init__(feature_dimension)
        self._ordered_parameters = ParameterVector(f"{block_type}_Amp_Encoder_Params_{hex(id(super()))}")


class AmplitudeEncoder(NonGateBlock, EncodingBlock):
    """
    Encode input as initial state of the circuit.
    """

    def __init__(self, block_type: Literal['input'] | Literal['weight'] = 'input') -> None:
        super().__init__("AmplitudeEncoder", block_type)

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        circuit = TypedRawFeatureVector(2**num_qubits, self.block_type)
        self._parameters = list(circuit.parameters)
        return circuit
