""" Formatter implementation translating for circuit parametrization. """
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from quantum_launcher.base.adapter_structure import formatter
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info.states.statevector import Statevector

from .problem import CircuitProblem


@formatter(CircuitProblem, 'none')
class _CircuitForwardFormatter:
    """ Formatter for CircuitProblem. """

    def __call__(self, problem: CircuitProblem, parameters: Mapping[Parameter, Iterable | complex] | Iterable | None = None,
                 auto_bind: bool = True, initial_state: Statevector | Sequence[complex] | str | int | None = None
                 ) -> list[tuple[QuantumCircuit, Iterable] | QuantumCircuit]:
        """Formatter for passing circuit and parameters, optionally can bind them

        Args:
            problem (CircuitProblem): Problem instance
            parameters (Mapping[Parameter, Iterable  |  complex] | Iterable | None): parameters for given circuit.
            auto_bind (bool, optional): If enabled binding is done before passing circuit to sampler. Defaults to True.
            initial_state (Statevector | Sequence[complex] | str | int | None, optional): Initial state for circuit. Defaults to None.

        Returns:
            list[tuple[QuantumCircuit, Iterable] | QuantumCircuit]: Qiskit pubs with either circuit or circuit and it's parameters.


        ### !Note: if auto binding enabled Mapping[Parameter, Iterable] doesn't work for single value Parameters.
        """
        if auto_bind:
            circuit = self.bind_params(problem, parameters, initial_state)
            return [circuit]
        if parameters is None:
            return [problem.instance]
        return [(problem.instance, parameters)]

    def bind_params(self, problem: CircuitProblem, parameters: Mapping[Parameter, Iterable] | Iterable | None,
                    initial_state: Statevector | Sequence[complex] | str | int | None) -> QuantumCircuit:
        """ Binding parameters """
        if initial_state is not None:
            old_circuit: QuantumCircuit = problem.instance
            circuit = QuantumCircuit(old_circuit.qubits)
            circuit.prepare_state(initial_state, normalize=True)
            circuit.compose(old_circuit, inplace=True)
        else:
            circuit: QuantumCircuit = problem.instance
        if parameters is None:
            return deepcopy(circuit)
        return circuit.assign_parameters(parameters)
