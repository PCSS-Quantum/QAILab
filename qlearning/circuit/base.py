"""ABC structure for circuit building blocks"""
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class CircuitBlock(ABC):
    """
    Base class for any circuit building block

    Attributes:
        num_qubits (int): Number of qubits for the block circuit.
        name (str): Block (and block circuit) name.
        circuit (QuantumCircuit): Block circuit.
    """

    _circuit: QuantumCircuit

    def __init__(self, num_qubits: int, name: str = 'unknown') -> None:
        self.num_qubits = num_qubits
        self.name = name
        self._circuit = QuantumCircuit(num_qubits, name=self.name)
        self._build_circuit()

    @property
    def circuit(self) -> QuantumCircuit:
        """Block circuit."""
        return self._circuit

    @abstractmethod
    def _build_circuit(self):
        pass

    @abstractmethod
    def add_to_circuit(self, circuit: QuantumCircuit):
        """
        Add this block to a circuit (number of qubits must match)

        Args:
            circuit (QuantumCircuit): The circuit that will receive this block (in place).
        """


class LayerBlock(CircuitBlock, ABC):
    """Weight, entangling, dropout etc."""

    def add_to_circuit(self, circuit: QuantumCircuit):
        circuit.append(self._circuit.to_gate(label=self.name), list(range(self.num_qubits)))


class WeightBlock(LayerBlock, ABC):
    """
    Parameterized blocks encoding nn weights.

    Attributes:
        weights (ParameterVector): Weights encoded by the block.
    """

    def __init__(self, num_qubits: int, name: str = 'BaseWeightBlock') -> None:
        self.weights = ParameterVector(f'weights_{hex(id(super()))}', num_qubits)
        super().__init__(num_qubits, name)


class EntanglingBlock(LayerBlock, ABC):
    """Blocks entangling qubits together"""


class EncodingBlock(CircuitBlock, ABC):
    """
    Blocks encoding the input vector

    Attributes:
        input (ParameterVector): Input vector encoded by the block.
    """

    def __init__(self, num_qubits: int, name: str = 'BaseEncodingBlock') -> None:
        self.input = ParameterVector(f'x_vector_{hex(id(super()))}', num_qubits)
        super().__init__(num_qubits, name)

    def add_to_circuit(self, circuit: QuantumCircuit):
        circuit.append(self._circuit.to_gate(label=self.name), list(range(self.num_qubits)))


class MeasurementBlock(CircuitBlock, ABC):
    """Blocks defining the output structure"""

    def add_to_circuit(self, circuit: QuantumCircuit):
        circuit.compose(self._circuit, list(range(self.num_qubits)), inplace=True)
