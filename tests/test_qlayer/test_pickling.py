""" Test whether the test is picklable. """
import pickle
import pytest
import torch
import qiskit.circuit
from qlearning.torch import QLayer


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit with encoding input as a params """
    weight_param = qiskit.circuit.ParameterVector('weight', 2)
    input_param = qiskit.circuit.ParameterVector('input_param', 2)
    circuit = qiskit.QuantumCircuit(2, 2)
    for i, param in enumerate(input_param):
        circuit.rx(param, i)
    for i, param in enumerate(weight_param):
        circuit.ry(param, i)
    circuit.measure(range(2), range(2))
    return circuit


def test_pickling_load_and_save():
    """ Test if QLayer can be pickled """
    model = QLayer(build_circuit())
    model_in_str = pickle.dumps(model)
    assert isinstance(model_in_str, bytes)
    new_model = pickle.loads(model_in_str)
    assert isinstance(new_model, QLayer)
    result = new_model(torch.Tensor([0, 1]))
    assert isinstance(result, torch.Tensor)


def test_pickling_load_and_save_after_forward_pass():
    """ Test if QLayer can be pickled after forward pass """
    model = QLayer(build_circuit())
    model(torch.Tensor([0, 1]))
    model_in_str = pickle.dumps(model)
    assert isinstance(model_in_str, bytes)
    new_model = pickle.loads(model_in_str)
    assert isinstance(new_model, QLayer)
    result = new_model(torch.Tensor([0, 1]))
    assert isinstance(result, torch.Tensor)
