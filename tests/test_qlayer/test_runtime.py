""" Base tests for QLayer """
import torch
from torch import nn
import qiskit
import qiskit.circuit

from qlearning.torch.qlayer import QLayer, ExpQLayer
from qlearning.torch.qmodel import QModel


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit """
    param = qiskit.circuit.ParameterVector('weight', 1)
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.rx(param[0], 0)
    circuit.measure(0, 0)
    return circuit


def build_circuit_param_based_input() -> qiskit.QuantumCircuit:
    """ build example circuit with encoding input as a params """
    param = qiskit.circuit.ParameterVector('weight', 2)
    input_param = qiskit.circuit.ParameterVector('input_param', 2)
    circuit = qiskit.QuantumCircuit(2, 2)
    circuit.rx(input_param[0], 0)
    circuit.rx(input_param[1], 1)
    circuit.ry(param[0], 0)
    circuit.ry(param[1], 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_runtime():
    """ Runtime test """
    quantum_model = QModel(layers = [QLayer(build_circuit())],loss=nn.MSELoss())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([1])
    test_input = torch.Tensor([-0.11, .3, ])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)


def test_integration():
    """ Integration with classical layers test """
    quantum_model = QModel(layers=[
        nn.Linear(4,2),
        QLayer(build_circuit()),
        nn.Linear(1, 1),
    ],loss = nn.MSELoss())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([1])
    test_input = torch.Tensor([-0.111111, .3, 1, 1])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)


def test_parameter_input_encoding():
    """ Testing if encoding input via parameters works properly """
    quantum_model = QModel(layers=[
        nn.Linear(4, 2),
        QLayer(build_circuit_param_based_input()),
        nn.Linear(2, 1),
    ],loss=nn.MSELoss())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([-0.111111, .3, 1, 1])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)


def test_exp_layer():
    """ Testing if encoding input via parameters works properly """
    quantum_model = QModel(layers=[
        nn.Linear(4, 2),
        ExpQLayer(build_circuit_param_based_input()),
        nn.Linear(4, 1),
    ],loss=nn.MSELoss())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([-0.321, .31, 0.3, 2])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)
