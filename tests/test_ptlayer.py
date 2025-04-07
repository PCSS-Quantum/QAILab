from qlearning.torch.qmodel import QModel
from qlearning.torch.qlayer import QLayer
from qlearning.torch.orca_layer import ORCALayer
from ptseries.models.pt_layer import PTLayer
import qiskit.circuit
import torch.nn as nn
import torch
from torch.optim.adam import Adam
from sklearn import datasets


def build_circuit() -> qiskit.QuantumCircuit:
    """ build example circuit with encoding input as a params """
    param = qiskit.circuit.ParameterVector('weight', 3)
    input_param = qiskit.circuit.ParameterVector('input_param', 3)
    circuit = qiskit.QuantumCircuit(3, 3)
    circuit.rx(input_param[0], 0)
    circuit.rx(input_param[1], 1)
    circuit.rx(input_param[2], 2)
    circuit.ry(param[0], 0)
    circuit.ry(param[1], 1)
    circuit.ry(param[2], 2)
    circuit.measure([0, 1, 2], [0, 1, 2])
    return circuit


def not_test_layers():
    """ Test integration with other models """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        ORCALayer(6, n_loops=2),
        nn.Linear(6, 3),
        #QLayer(build_circuit()),
        #nn.Linear(3, 3),
        nn.Softmax()

    ), nn.CrossEntropyLoss(), optimizer_type=Adam, batch_size=10, epochs=20)
    iris = datasets.load_iris()
    data, target = iris.data, iris.target
    result = torch.argmax(q_model.fit_predict(data, target), dim=1)
    assert result.shape == target.shape
    print(result == target)


if __name__ == "__main__":
    not_test_layers()
