""" Formatter implementation translating for circuit parametrization. """
from quantum_launcher.base.adapter_structure import formatter
from qiskit import QuantumCircuit

from .problem import CircuitProblem


@formatter(CircuitProblem, 'none')
class _CircuitFormatter:
    """ Formatter for CircuitProblem. """

    def __call__(self, problem: CircuitProblem, weights: dict) -> tuple[QuantumCircuit, dict]:
        """ Checking if arguments are proper. """

        return problem.instance, weights
