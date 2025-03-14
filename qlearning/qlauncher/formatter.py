""" Formatter implementation translating for circuit parametrization. """
from copy import deepcopy
from collections.abc import Iterable
from quantum_launcher.base.adapter_structure import formatter
from qiskit import QuantumCircuit

from .problem import CircuitProblem


@formatter(CircuitProblem, 'none')
class _CircuitFormatter:
    """ Formatter for CircuitProblem. """

    def __call__(self, problem: CircuitProblem, weights: Iterable,
                 auto_bind: bool = False) -> list[tuple[QuantumCircuit, Iterable] | QuantumCircuit]:
        """ Checking if arguments are proper. """
        if auto_bind:
            circuit = self.bind_params(problem, weights)
            return [circuit]
        return [(problem.instance, weights)]

    def bind_params(self, problem: CircuitProblem, weights: Iterable) -> QuantumCircuit:
        """ Binding parameters """
        circuit: QuantumCircuit = deepcopy(problem.instance)
        print(circuit.parameters)
        return circuit.assign_parameters(weights)
