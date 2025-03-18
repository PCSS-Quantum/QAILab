""" Formatter implementation translating for circuit parametrization. """
from copy import deepcopy
from collections.abc import Iterable, Mapping
from quantum_launcher.base.adapter_structure import formatter
from qiskit.circuit import QuantumCircuit, Parameter

from .problem import CircuitProblem


@formatter(CircuitProblem, 'none')
class _CircuitForwardFormatter:
    """ Formatter for CircuitProblem. """

    def __call__(self, problem: CircuitProblem, parameters: Mapping[Parameter, Iterable] | Iterable,
                 auto_bind: bool = False) -> list[tuple[QuantumCircuit, Iterable] | QuantumCircuit]:
        """ Checking if arguments are proper. """
        if auto_bind:
            circuit = self.bind_params(problem, parameters)
            return [circuit]
        return [(problem.instance, parameters)]

    def bind_params(self, problem: CircuitProblem, parameters: Mapping[Parameter, Iterable] | Iterable) -> QuantumCircuit:
        """ Binding parameters """
        circuit: QuantumCircuit = deepcopy(problem.instance)
        return circuit.assign_parameters(parameters)
