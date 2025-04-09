from qlearning.torch.qmodel import QModel
from qlearning.torch.qlayer import QLayer
from qlearning.torch.orca_layer import ORCALayer
from ptseries.optimizers import HybridOptimizer
import qiskit.circuit
import torch.nn as nn
import torch
from torch.optim.adam import Adam
from sklearn import datasets


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit with encoding input as a params """
    param = qiskit.circuit.ParameterVector('weight-P', 4)
    input_param = qiskit.circuit.ParameterVector('input_param', 4)
    circuit = qiskit.QuantumCircuit(4, 4)
    circuit.rx(input_param[0], 0)
    circuit.rx(input_param[1], 1)
    circuit.rx(input_param[2], 2)
    circuit.rx(input_param[3], 3)
    circuit.ry(param[0], 0)
    circuit.ry(param[1], 1)
    circuit.ry(param[2], 2)
    circuit.ry(param[3], 3)
    circuit.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return circuit


def test_orca_layer():
    """ Test orca layer """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        ORCALayer(6, n_loops=2),
        nn.Linear(6, 3),
        nn.Softmax()

    ), nn.CrossEntropyLoss(), optimizer_type=Adam, epochs=1)
    iris = datasets.load_iris()
    data, target = iris.data, iris.target
    result = torch.argmax(q_model.fit_predict(data, target), dim=1)
    assert result.shape == target.shape


def test_quantum_layers():
    """ Test quantum layers working together """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        ORCALayer(4, n_loops=2),
        nn.Linear(4, 4),
        nn.ReLU(),
        QLayer(build_circuit()),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Softmax()
    ), nn.CrossEntropyLoss(), optimizer_type=HybridOptimizer, quantum_learning_rate=0.05, batch_size=10, epochs=50, metric="mse")
    iris = datasets.load_iris()
    data, target = iris.data, iris.target
    result = torch.argmax(q_model.fit_predict(data, target), dim=1)
    assert result.shape == target.shape


if __name__ == "__main__":
    test_quantum_layers()
