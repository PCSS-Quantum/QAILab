"""R-gate implementations of input vector encoding blocks for variational circuits."""
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qlearning.circuit.base import EncodingBlock


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
        if self._parameter_vector is None:
            self._parameter_vector = ParameterVector(f"{self.block_type}_R_Encoder_Params_{hex(id(super()))}", num_qubits)

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
            fn(self._parameter_vector[i], i)

        return circuit
