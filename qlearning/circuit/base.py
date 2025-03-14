from abc import ABC  # , abstractmethod
# from typing import override

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class CircuitBlock(ABC):

    _circuit: QuantumCircuit

    def __init__(self, num_qubits, name='unknown') -> None:
        self.num_qubits = num_qubits
        self.name = name

    @property
    def qiskit_gate(self):
        return self._circuit.to_gate(label=self.name)


class WeightBlock(CircuitBlock, ABC):
    def __init__(self, num_qubits, name='BaseWeightBlock') -> None:
        super().__init__(num_qubits, name)
        self.weights_pv = ParameterVector(f'weights_{hex(id(super()))}', num_qubits)


class EncodingBlock(CircuitBlock, ABC):
    def __init__(self, num_qubits, name='BaseEncodingBlock') -> None:
        super().__init__(num_qubits, name)
        self.x_pv = ParameterVector(f'x_vector_{hex(id(super()))}', num_qubits)


class EntanglingBlock(CircuitBlock):
    pass


class MeasurementBlock(CircuitBlock, ABC):
    pass
