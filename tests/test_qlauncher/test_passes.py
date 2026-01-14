"""Forward pass tests"""

import math

import pytest

from qlauncher import QLauncher, Result
from qlauncher.routines.qiskit import QiskitBackend, AQTBackend
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qailab.qlauncher import NNCircuit, ForwardPass


def _trainable_circuit() -> tuple[NNCircuit, list[Parameter]]:
    input_param = Parameter("input")
    weight = Parameter("weights")
    circuit = QuantumCircuit(1, 1)
    circuit.rx(input_param, 0)
    circuit.rz(weight, 0)
    circuit.measure(0, 0)
    return NNCircuit(circuit, "test"), [input_param, weight]


def test_forward_pass_runtime():
    """Tests basic forward pass runtime."""
    problem, _ = _trainable_circuit()
    algorithm = ForwardPass(shots=1)
    launcher = QLauncher(problem, algorithm, QiskitBackend("local_simulator"))
    results = launcher.run(parameters=[0, 0], initial_state=[0, 1])
    assert isinstance(results, Result)
    assert isinstance(results.distribution, dict)
    assert results.distribution[(1,)] == 1
    assert results.num_of_samples == 1


def test_forward_pass_without_params():
    """Tests basic forward pass runtime."""
    problem, _ = _trainable_circuit()
    problem.instance.assign_parameters([0, 0], inplace=True)
    algorithm = ForwardPass(shots=1)
    launcher = QLauncher(problem, algorithm, QiskitBackend("local_simulator"))
    results = launcher.run(initial_state=[0, 1])
    assert isinstance(results, Result)
    assert isinstance(results.distribution, dict)
    assert results.distribution[(1,)] == 1
    assert results.num_of_samples == 1


def test_forward_pass_runtime_auto_bind():
    """Tests basic forward pass runtime."""
    problem, _ = _trainable_circuit()
    algorithm = ForwardPass(shots=1)
    launcher = QLauncher(problem, algorithm, QiskitBackend("local_simulator"))
    results = launcher.run(parameters=[1, 1])
    assert isinstance(results, Result)
    assert isinstance(results.distribution, dict)
    assert results.num_of_samples == 1


def test_forward_pass_weight_assignment():
    """Test if forward pass weights assignment works."""
    problem, _ = _trainable_circuit()
    algorithm = ForwardPass(shots=5)
    launcher = QLauncher(problem, algorithm, QiskitBackend("local_simulator"))
    results = launcher.run(parameters=[0, 0])
    assert results.distribution[(0,)] == 1
    results = launcher.run(parameters=[math.pi, math.pi])
    assert results.distribution[(1,)] == 1
    results = launcher.run(parameters=[math.pi, math.pi])
    assert results.distribution[(1,)] == 1


def test_forward_pass_weight_assignment_by_dict():
    """Test if forward pass weights assignment works."""
    problem, (input_param, weight) = _trainable_circuit()
    algorithm = ForwardPass(shots=5)
    launcher = QLauncher(problem, algorithm, QiskitBackend("local_simulator"))
    results = launcher.run(parameters={input_param: 0, weight: 0})
    assert results.distribution[(0,)] == 1
    results = launcher.run(parameters={input_param: math.pi, weight: math.pi})
    assert results.distribution[(1,)] == 1


@pytest.mark.skip("qlauncher issue")
def test_forward_pass_auto_assignment():
    """Test if forward pass auto parameter assignment works. auto_bind is required for AQT"""
    problem, _ = _trainable_circuit()
    algorithm = ForwardPass(shots=5)
    launcher = QLauncher(problem, algorithm, AQTBackend("local_simulator"))
    results = launcher.run(parameters=[0, 0])
    assert results.distribution[(0,)] == 1
    results = launcher.run(parameters=[math.pi, math.pi])
