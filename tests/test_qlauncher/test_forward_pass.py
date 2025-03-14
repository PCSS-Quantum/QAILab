""" Forward pass tests """
from quantum_launcher import Result
from quantum_launcher.base.adapter_structure import get_formatter
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qlearning.qlauncher.passes.forward import ForwardPass
from qlearning.qlauncher.problem import CircuitProblem
from qlearning.qlauncher.formatter import _CircuitFormatter


def _trainable_circuit() -> QuantumCircuit:
    weight = Parameter('weight')
    circuit = QuantumCircuit(1, 1)
    circuit.rx(weight, 0)
    circuit.measure(0, 0)
    return circuit


def test_runtime():
    """ Tests basic runtime. """
    assert _CircuitFormatter is not None
    problem = CircuitProblem(_trainable_circuit(), 'test')
    algorithm = ForwardPass(5)
    formatter = get_formatter(CircuitProblem, 'none')
    formatter.set_run_params(params={'weights': 1})
    results = algorithm.run(problem, QiskitBackend('local_simulator'), formatter=formatter)
    assert isinstance(results, Result)
    print()
    print(results.distribution)
    print(f'{'.':=^20}')
