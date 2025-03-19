""" Module with QModel """
from collections.abc import Callable
from torch import Tensor
import torch
from torch.nn import MSELoss, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator

from qlearning.torch.qlayer import QLayer


class QModel(Module, BaseEstimator):
    """ Quantum model class """
    layers: list[Module]
    optimizer: Optimizer
    loss: Callable
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool

    def __init__(
        self,
        layers: list[Module] | None = None,
        optimizer: Optimizer | None = None,
        loss: Callable | None = None,
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True
    ):
        super().__init__()
        if layers is None:
            layers = []
        self.layers = layers
        if optimizer is None:
            optimizer = Adam(self.parameters())
        self.optimizer = optimizer
        if loss is None:
            loss = MSELoss()
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle

    def reset_parameters(self) -> None:
        """ Resets parameters of QLayers """
        for layer in self.layers:
            if isinstance(layer, QLayer):
                layer.reset_parameters()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """ Forward """
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    def fit(self, x: Tensor, y: Tensor,) -> "QModel":
        """ scikit-learn like fit method """
        if x.shape[0] != y.shape[0]:
            raise ValueError("X i y tensors should have the same first dimension")
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        return self

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        for _ in tqdm(range(epochs), total=epochs, unit="epochs"):

            self.train()
            _ = self._train_one_epoch(train_loader)

            self.eval()
            with torch.no_grad():
                _ = self._validate_one_epoch(validation_loader)

    def _train_one_epoch(self, train_loader: DataLoader) -> np.floating:
        losses = []

        for (x, y) in tqdm(train_loader, unit="batches"):
            self.optimizer.zero_grad()
            outputs = self(x)
            loss = self.loss(outputs, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> np.floating:
        losses = []

        for (x, y) in tqdm(validation_loader, unit="batches"):
            outputs = self(x)
            loss = self.loss(outputs, y)
            losses.append(loss.item())

        return np.mean(losses)

    def predict(self, x: Tensor):
        """ scikit-learn like predict method """
        return self(x)
