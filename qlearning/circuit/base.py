"""ABC structure for circuit building blocks"""
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.gate import Gate


class CircuitBlock(ABC):
    """
    Base class for any circuit building block

    Attributes:
        num_qubits (int): Number of qubits for the block circuit.
        name (str): Block (and block circuit) name.
        circuit (QuantumCircuit): Block circuit.
    """

    _circuit: QuantumCircuit

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

    def add_to_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Add this block to a circuit (number of qubits must match)

        Args:
            circuit (QuantumCircuit): The circuit that will receive this block (in place).
        """
        gate = self.to_gate(circuit.num_qubits)
        qargs = list(range(circuit.num_qubits))

        circuit.append(gate, qargs)

    @abstractmethod
    def _build_circuit(self, num_qubits: int) -> QuantumCircuit:
        pass


class ParameterizedBlock(CircuitBlock, ABC):
    """Blocks generating parametrized circuits"""

    def __init__(self, name: str = 'unknown') -> None:
        self._parameter_vector = None
        super().__init__(name)

    @property
    def parameter_vector(self) -> ParameterVector:
        """Get a parameter vector"""
        if self._parameter_vector is None:
            raise ValueError("No parameter vector, the block was not added to any circuit.")
        return self._parameter_vector


class EntanglingBlock(CircuitBlock, ABC):
    """Blocks entangling qubits together"""


class EncodingBlock(ParameterizedBlock, CircuitBlock, ABC):
    """Blocks encoding some parameter vector (trainable or not)"""


class MeasurementBlock(CircuitBlock, ABC):
    """Blocks defining the output structure"""

    def add_to_circuit(self, circuit: QuantumCircuit) -> None:
        c = self._build_circuit(circuit.num_qubits)
        qargs = list(range(circuit.num_qubits))
        circuit.compose(c, qargs, inplace=True)
