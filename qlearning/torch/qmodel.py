""" Module with QModel """
from collections.abc import Callable
from typing import Literal
import torch
from torch import Tensor, optim
from torch.nn import MSELoss, Module, Linear, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator

from qlearning.torch.qlayer import QLayer

available_optimizers: dict[str, type[Optimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad, optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS]}


class QModel(Module, BaseEstimator):
    """ Quantum model class """
    layers: ModuleList
    optimizer: Optimizer
    loss: Callable
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool

    def __init__(
        self,
        layers: list[Module],
        optimizer: type[Optimizer] | str | None = None,
        loss: Callable | None = None,
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu"
    ):
        super().__init__()
        self.layers = ModuleList(layers)
        if optimizer is None:
            optimizer = optim.AdamW
        if isinstance(optimizer, str):
            try:
                optimizer = available_optimizers[optimizer]
            except ValueError as e:
                raise ValueError(
                    f"Unknown optimizer: {optimizer}. Available optimizers are: {list(available_optimizers.keys())}") from e
        self.optimizer = optimizer(self.parameters())
        if loss is None:
            loss = MSELoss()
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.device = device
        self.to(device)

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

    def fit(self, x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> "QModel":
        """ scikit-learn like fit method """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y tensors should have the same first dimension")
        x, y = x.to(self.device), y.to(self.device)
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        return self

    def fit_predict(self, x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> Tensor:
        """ scikit-learn like fit_predict method """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y tensors should have the same first dimension")
        x, y = x.to(self.device), y.to(self.device)
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        self.eval()
        with torch.inference_mode():
            return self(x).cpu()

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        pbar = tqdm(range(epochs), total=epochs, unit="epochs")
        for epoch in pbar:

            self.train()
            _ = self._train_one_epoch(train_loader)

            self.eval()
            with torch.inference_mode():
                valid_loss = self._validate_one_epoch(validation_loader)
            pbar.set_postfix(loss=valid_loss, epoch=epoch+1)

    def _train_one_epoch(self, train_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(train_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            outputs = self(x)
            loss = self.loss(outputs, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch+1)

        return np.mean(losses)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(validation_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            outputs = self(x)
            loss = self.loss(outputs, y)
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch+1)

        return np.mean(losses)

    def predict(self, x: Tensor | np.ndarray) -> Tensor:
        """ scikit-learn like predict method """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        self.eval()
        with torch.inference_mode():
            return self(x).cpu()


