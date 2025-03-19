""" Base tests for QLayer """
import torch
from torch import nn
import qiskit
import qiskit.circuit

from qlearning.torch.qlayer import QLayer


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit """
    param = qiskit.circuit.ParameterVector('weight', 1)
    circuit = qiskit.QuantumCircuit(1, 1)
    circuit.rx(param[0], 0)
    circuit.measure(0, 0)
    return circuit


class QuantumModel(nn.Module):
    """ QModel """

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            QLayer(build_circuit()),
        )

    def forward(self, x):
        """ Forward pass """
        return self.net(x)


def test_runtime():
    """ Runtime test """
    quantum_model = QuantumModel()
    loss_fn = nn.MSELoss()
    desired_result = torch.Tensor([1])
    predictions = quantum_model(torch.Tensor([0.111111, .3]))
    assert isinstance(predictions, torch.Tensor)
    loss = loss_fn(predictions, desired_result)
    assert isinstance(loss, torch.Tensor)
