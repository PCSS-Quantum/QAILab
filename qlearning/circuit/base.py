"""ABC structure for circuit building blocks"""
from abc import ABC, abstractmethod
from typing import Literal
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QubitSpecifier


class CircuitBlock(ABC):
    """
    Base class for any circuit building block

    Attributes:
        name (str): Block (and block circuit) name.
    """

    def __init__(self, name: str = 'unknown') -> None:
        self.name = name

    def to_gate(self, num_qubits: int) -> Gate:
        """
        Get a gate form of of this block.

        Args:
            num_qubits (int): Desired width.

        Returns:
            Gate: Block defined circuit as a single gate.
        """
        qc = self._build_circuit(num_qubits)
        return qc.to_gate(label=self.name)

    def add_to_circuit(self, circuit: QuantumCircuit, qargs: Sequence[QubitSpecifier] | None = None) -> None:
        """
        Add this block to a circuit (number of qubits must match)

        Args:
            circuit (QuantumCircuit): The circuit that will receive this block (in place).
            qargs (list[QubitSpecifier] | None, optional): Which qubits to apply this circuit to. If None apply to all. Defaults to None.
        """
        if qargs is None:
            qargs = list(range(circuit.num_qubits))
        else:
            qargs = list(qargs)

        gate = self.to_gate(len(qargs))

        if gate.num_qubits > len(qargs):
            num_qb_before = circuit.num_qubits
            circuit.add_register(QuantumRegister(gate.num_qubits - len(qargs)))
            qargs += list(range(num_qb_before, circuit.num_qubits))

        circuit.append(gate, qargs)

    @abstractmethod
    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        pass


class ParameterizedBlock(CircuitBlock, ABC):
    """
    Blocks generating parametrized circuits
    """

    def __init__(self, name: str = 'unknown') -> None:
        self._parameters: Sequence[Parameter] | None = None
        super().__init__(name)

    @property
    def parameters(self) -> Sequence[Parameter]:
        """Get this block's parameter vector"""
        if self._parameters is None:
            raise ValueError("No parameters, the block was not added to any circuit.")
        return self._parameters


class EntanglingBlock(CircuitBlock, ABC):
    """Blocks entangling qubits together"""


class EncodingBlock(ParameterizedBlock, CircuitBlock, ABC):
    """
    Blocks encoding some parameter vector (trainable or not)

    Attributes:
        block_type (Literal['input', 'weight']): Whether this block encodes weights or inputs.
    """

    def __init__(self, name: str = 'unknown', block_type: Literal['input', 'weight'] = 'input') -> None:
        self.block_type = block_type
        super().__init__(name)

    def _create_parameters(self, size: int) -> Sequence[Parameter]:
        parameter_vector = ParameterVector(f"{self.block_type}_{self.__class__.__name__}_Params_{hex(id(super()))}", size)
        return parameter_vector.params


class NonGateBlock(CircuitBlock, ABC):
    """Blocks that cannot be converted to gates, e.g. measurement."""

    def to_gate(self, num_qubits: int) -> Gate:
        raise ValueError("This block cannot be converted to a gate.")

    def add_to_circuit(self, circuit: QuantumCircuit, qargs: Sequence[QubitSpecifier] | None = None) -> None:
        if qargs is None:
            qargs = list(range(circuit.num_qubits))
        else:
            qargs = list(qargs)

        c = self._build_circuit(len(qargs))

        circuit.compose(c, qargs, inplace=True)
