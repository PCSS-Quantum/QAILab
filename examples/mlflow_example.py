""" Example of integration QLearning with MLFlow """
# Requires installation of MLFlow
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.optim import Adam
import qiskit
import qiskit.circuit
import mlflow
import mlflow.pytorch
from qlearning.torch.qlayer import QLayer


def create_dataset() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates circle dataset

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train_x, train_y, predict_x, predict_y
    """
    train_x = (np.random.rand(25, 2) * 2) - 1
    train_y = np.array([float(x**2 + y**2 < 1) for (x, y) in train_x])
    predict_x = (np.random.rand(10, 2) * 2) - 1
    predict_y = np.array([float(x**2 + y**2 < 1) for (x, y) in predict_x])
    return Tensor(train_x), Tensor(train_y), Tensor(predict_x), Tensor(predict_y)


def create_model() -> nn.Module:
    """ build example circuit with encoding input as a params """
    weight_param = qiskit.circuit.ParameterVector('weight', 2)
    input_param = qiskit.circuit.ParameterVector('input_param', 2)
    circuit = qiskit.QuantumCircuit(2, 2)
    for i, param in enumerate(input_param):
        circuit.rx(param, i)
    for i, param in enumerate(weight_param):
        circuit.ry(param, i)
    circuit.measure(range(2), range(2))

    return torch.nn.Sequential(
        QLayer(circuit),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )


def training_loop(module: nn.Module, train_x: Tensor, train_y: Tensor):
    """ Basic training loop for model """
    epochs = 100
    optimizer = Adam(module.parameters(), lr=0.01)
    loss = nn.MSELoss()
    for _ in tqdm(range(epochs)):
        for x, y in zip(train_x, train_y):
            module.train()
            optimizer.zero_grad()
            outputs = module(x)
            current_loss = loss(outputs, y)
            current_loss.backward()
            optimizer.step()


def main():
    """ main """
    with mlflow.start_run() as run:
        model = create_model()
        train_x, train_y, pred_x, _ = create_dataset()
        training_loop(model, train_x, train_y)

        mlflow.pytorch.log_model(model, artifact_path="model")
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

    new_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    print('Model loaded!!!')
    prediction = new_model(pred_x)
    print("Predicted:", prediction)


if __name__ == '__main__':
    main()
