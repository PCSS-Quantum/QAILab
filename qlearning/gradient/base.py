""" File for gradient calculation structure """
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2


class GradientCalculation(ABC):
    """ Base class for calculating gradient """

    def __init__(self, learning_rate: float = 0.1) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def calculate_gradient(self, sampler: BaseSamplerV2,
                           pub: list[QuantumCircuit | tuple[QuantumCircuit, list]]) -> list[float]:
        """Calculates gradient of quantum circuits

        Args:
            sampler (BaseSamplerV2): sampler
            pub (list[QuantumCircuit  |  tuple[QuantumCircuit, list]]): list of quantum circuits
        """
