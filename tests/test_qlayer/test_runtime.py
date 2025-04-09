""" Base tests for QLayer """
import torch
from torch import nn
import qiskit
import qiskit.circuit

from qlearning.torch import QLayer, ExpectedValueQLayer, ArgmaxQLayer


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit """
    inpt = qiskit.circuit.ParameterVector('input', 2)
    param = qiskit.circuit.ParameterVector('weight', 1)
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.rx(param[0], 0)
    circuit.ry(inpt[0], 0)
    circuit.ry(inpt[1], 0)
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


def make_qmodel(qlayer_type: type[QLayer], circ) -> nn.Module:
    class QuantumModel(nn.Module):
        """ Hybrid model """

        def __init__(self):
            super().__init__()
            ql = qlayer_type(circ)
            self.net = torch.nn.Sequential(
                nn.Linear(4, ql.in_features),
                ql,
                nn.Linear(ql.out_features, 1),
            )

        def forward(self, x):
            """ Forward pass """
            return self.net(x)

    return QuantumModel()


def test_runtime():
    """ Runtime test """
    quantum_model = make_qmodel(QLayer, build_circuit())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([-0.111111, .3, 1, 1])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)

    loss.backward()  # Test if calling backward generates no errors


def test_parameter_input_encoding():
    """ Testing if encoding input via parameters works properly """

    quantum_model = make_qmodel(QLayer, build_circuit_param_based_input())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([-0.111111, .3, 1, 1])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)

    loss.backward()  # Test if calling backward generates no errors


def test_expected_value_layer():
    """ Testing if encoding input via parameters works properly """
    quantum_model = make_qmodel(ExpectedValueQLayer, build_circuit_param_based_input())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([[-0.321, .31, 0.3, 2]])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)

    loss.backward()  # Test if calling backward generates no errors

    ql: ExpectedValueQLayer = quantum_model.net[1]
    assert ql._max_expected_out_value == 2**ql.circuit.num_clbits - 1

    ql._rescale_output_range = (-1, 1)
    assert ql._rescale_out(torch.tensor([0])) == torch.tensor([-1])


def test_argmax_layer():
    quantum_model = make_qmodel(ArgmaxQLayer, build_circuit_param_based_input())
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([[-0.321, .31, 0.3, 2]])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)

    loss.backward()  # Test if calling backward generates no errors


def test_unweighted_qlayer():
    """Test if layers with no weights work as normal"""
    circ = qiskit.circuit.QuantumCircuit(2, 2)
    p1, p2 = qiskit.circuit.Parameter('input1'), qiskit.circuit.Parameter('input2')
    circ.rx(p1, 0)
    circ.rx(p2, 1)
    circ.measure([0, 1], [0, 1])

    quantum_model = make_qmodel(QLayer, circ)
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([0, 1])
    test_input = torch.Tensor([[-0.321, .31, 0.3, 2]])
    predictions = quantum_model(test_input)
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)

    loss.backward()  # Test if calling backward generates no errors
