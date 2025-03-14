""" Forward pass tests """
import math

from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.routines.qiskit_routines import QiskitBackend
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qlearning.qlauncher import CircuitProblem, ForwardPass, BackwardPass


def _trainable_circuit() -> CircuitProblem:
    weight = Parameter('weight')
    circuit = QuantumCircuit(1, 1)
    circuit.rx(weight, 0)
    circuit.measure(0, 0)
    return CircuitProblem(circuit, 'test')


def test_forward_pass_runtime():
    """ Tests basic forward pass runtime. """
    problem = _trainable_circuit()
    algorithm = ForwardPass(shots=1)
    launcher = QuantumLauncher(problem, algorithm, QiskitBackend('local_simulator'))
    results = launcher.run(weights=[1])
    assert isinstance(results, Result)
    assert isinstance(results.distribution, dict)
    assert results.num_of_samples == 1


def test_forward_pass_weight_assignment():
    """ Test if forward pass weights assignment works. """
    problem = _trainable_circuit()
    algorithm = ForwardPass(shots=5)
    launcher = QuantumLauncher(problem, algorithm, QiskitBackend('local_simulator'))
    results = launcher.run(weights=[0])
    assert results.distribution[(0,)] == 1
    results = launcher.run(weights=[math.pi])
    assert results.distribution[(1,)] == 1


def test_backward_pass_runtime():
    """ Tests basic backward pass runtime. """
    problem = _trainable_circuit()
    algorithm = BackwardPass(shots=1)
    launcher = QuantumLauncher(problem, algorithm, QiskitBackend('local_simulator'))
    results = launcher.run(weights=[1])
    assert isinstance(results, Result)
    assert isinstance(results.distribution, dict)
    assert results.num_of_samples == 1


def test_backward_pass_weight_assignment():
    """ Test if backward pass weights assignment works. """
    problem = _trainable_circuit()
    algorithm = BackwardPass(shots=5)
    launcher = QuantumLauncher(problem, algorithm, QiskitBackend('local_simulator'))
    results = launcher.run(weights=[0])
    assert results.distribution[(0,)] == 1
    results = launcher.run(weights=[math.pi])
    assert results.distribution[(1,)] == 1
