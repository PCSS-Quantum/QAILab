"""QLauncher problem implementation"""

from collections.abc import Mapping, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info.states.statevector import Statevector
from qlauncher.base import Problem, ProblemLike


class ParameterizedCircuit(ProblemLike):
    """Problem class for not parametrized circuits"""

    def __init__(self, circuit: QuantumCircuit, parameter_values: Mapping[Parameter, float] | None) -> None:
        super().__init__((circuit, parameter_values))
        self._circuit = circuit
        self._parameter_values = parameter_values

    @property
    def bound_circuit(self) -> QuantumCircuit:
        return self._circuit.assign_parameters(self._parameter_values) if self._parameter_values is not None else self._circuit


class NNCircuit(Problem):
    def __init__(self, circuit: QuantumCircuit, instance_name: str = "unnamed-NNCircuit") -> None:
        super().__init__(circuit, instance_name)
        self._circuit = circuit

    def to_parameterized_circuit(
        self,
        parameters: Mapping[Parameter, float] | None = None,
        initial_state: Statevector | Sequence[complex] | str | int | None = None,
    ) -> ParameterizedCircuit:
        if parameters is None and len(self._circuit.parameters) > 0:
            raise ValueError("Please provide parameter values for the circuit.")

        if initial_state is not None:
            circuit = QuantumCircuit(self._circuit.qubits)
            circuit.prepare_state(initial_state, normalize=True)
            circuit.compose(self._circuit, inplace=True)
        else:
            circuit = self._circuit

        return ParameterizedCircuit(circuit, parameters)
