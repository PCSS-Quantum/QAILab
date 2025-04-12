"""R-gate implementations of input vector encoding blocks for variational circuits."""
from typing import Literal
from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

from qailab.circuit.base import EncodingBlock, EntanglingBlock


class RotationalEncoder(EncodingBlock):
    """
    Encoding of input vector using rotational gates.

    Attributes:
        r_gate_type (Literal['x', 'y', 'z']): Type of rotational gate applied to each qubit.
        block_type (Literal['input', 'weight']): Whether this block encodes weights or inputs.
    """

    def __init__(self, r_gate_type: Literal['x', 'y', 'z'], block_type: Literal['input', 'weight']) -> None:
        self.r_gate_type: Literal['x', 'y', 'z'] = r_gate_type
        super().__init__(f"R{r_gate_type}Encoder", block_type)

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        # Don't create new parameter vectors. This would enable the user to encode the same vector in different parts of the circuit.
        if self._parameters is None:
            self._parameters = self._create_parameters(num_qubits)

        circuit = QuantumCircuit(num_qubits)
        match self.r_gate_type:
            case 'x':
                fn = circuit.rx
            case 'y':
                fn = circuit.ry
            case 'z':
                fn = circuit.rz
            case _:
                raise ValueError(f"'{self.r_gate_type}' is not a valid rotational gate type")
        for i in range(num_qubits):
            fn(self._parameters[i], i)

        return circuit


class RealAmplitudesBlock(EncodingBlock, EntanglingBlock):
    """
    Block wrapper for RealAmplitudes from qiskit.circuit.library
    """

    def __init__(
        self,
        block_type: Literal['input', 'weight'],
        entanglement: str | list[list[int]] | Callable[[int], list[int]] = "reverse_linear",
        reps: int = 3,
        skip_final_rotation_layer: bool = False
    ) -> None:
        self.entanglement = entanglement
        self.reps = reps
        self.skip_final_rotation_layer = skip_final_rotation_layer
        super().__init__("RealAmplitudesEncoder", block_type)

    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        pv_name = self._create_parameters(1)[0].name
        amplitudes = RealAmplitudes(
            num_qubits,
            parameter_prefix=pv_name,
            entanglement=self.entanglement,
            reps=self.reps,
            skip_final_rotation_layer=self.skip_final_rotation_layer,
        )
        self._parameters = list(amplitudes.parameters)
        return amplitudes
