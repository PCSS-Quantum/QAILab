""" Module with QModel """
from collections.abc import Callable
from typing import Literal
import torch
from torch import Tensor, optim
from torch.nn import MSELoss, Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import pandas as pd

from qlearning.torch.qlayer import QLayer

available_optimizers: dict[str, type[Optimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad, optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS]}


class QModel(Module):
    """ Quantum model class """
    layers: ModuleList
    optimizer_type: type[Optimizer]
    optimizer: Optimizer
    learning_rate: int | None
    loss: Callable
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    def __init__(
        self,
        layers: list[Module],
        optimizer_type: type[Optimizer] | str | None = None,
        learning_rate: float | None = None,
        loss: Callable | None = None,
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu"
    ):
        super().__init__()
        self.layers = ModuleList(layers)
        if optimizer_type is None:
            optimizer_type = optim.AdamW
        if isinstance(optimizer_type, str):
            try:
                optimizer_type = available_optimizers[optimizer_type]
            except ValueError as e:
                raise ValueError(
                    f"Unknown optimizer: {optimizer_type}. Available optimizers are: {list(available_optimizers.keys())}") from e
        self.optimizer_type = optimizer_type
        if learning_rate is not None:
            self.optimizer = optimizer_type(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = self.optimizer_type(self.parameters())
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

    def fit(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> "QModel":
        """ scikit-learn like fit method """
        x, y = self._validate_x_y(x, y)
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        return self

    def _validate_x(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(self.device)
        return x

    def _validate_x_y(
        self,
        x: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame | pd.Series
    ) -> tuple[Tensor, Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            if y.dtype.kind == "i":
                y = torch.tensor(y, dtype=torch.int64)
            else:
                y = torch.tensor(y, dtype=torch.float32)
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        if isinstance(y, pd.DataFrame):
            y = torch.tensor(y.values)
        if isinstance(y, pd.Series):
            if y.dtype == np.dtype('int64'):
                y = torch.tensor(y.values, dtype=torch.int64)
            else:
                y = torch.tensor(y.values, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y tensors should have the same first dimension")
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def fit_predict(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> Tensor:
        """ scikit-learn like fit_predict method """
        x, y = self._validate_x_y(x, y)
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
            pbar.set_postfix(loss=valid_loss, epoch=epoch + 1)

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
            pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(validation_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            outputs = self(x)
            loss = self.loss(outputs, y)
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses)

    def predict(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        """ scikit-learn like predict method """
        x = self._validate_x(x)
        self.eval()
        with torch.inference_mode():
            return self(x).cpu()

    def get_params(self) -> dict:
        """" returns values of constructor parameters """
        return {
            "layers": self.layers,
            "optimizer_type": self.optimizer_type,
            "loss": self.loss,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_fraction": self.validation_fraction,
            "shuffle": self.shuffle,
            "device": self.device
        }

    def set_params(self, **params):
        """ scikit-learn like param setting method"""
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {valid_params.keys()!r}."
                )
            if key == "device":
                self.device = value
                self.to(self.device)
            if key == "optimizer_type":
                self.optimizer_type = value
                self.optimizer = self.optimizer_type(self.parameters())
            if key == "layers":
                self.layers = ModuleList(value)
        return self
