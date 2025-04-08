""" Module with QModel """
from collections.abc import Callable
from typing import Literal
from sklearn.base import BaseEstimator
import torch
from torch import Tensor, optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import pandas as pd

AVAILABLE_OPTIMIZERS: dict[str, type[Optimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad, optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS]}


class QModel(BaseEstimator):
    """ Quantum model class """
    module: nn.Module
    loss: Callable
    optimizer_type: type[Optimizer]
    optimizer: Optimizer
    learning_rate: float | Literal['auto']
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    def __init__(
        self,
        module: nn.Module,
        loss: Callable,
        optimizer_type: type[Optimizer] | str = 'adamw',
        learning_rate: float | Literal['auto'] = 'auto',
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu"
    ):
        super().__init__()
        self.module = module
        self.loss = loss
        if isinstance(optimizer_type, str):
            if optimizer_type not in AVAILABLE_OPTIMIZERS:
                raise ValueError(
                    f"Unknown optimizer: {optimizer_type}. Available optimizers are: {list(AVAILABLE_OPTIMIZERS.keys())}")
            optimizer_type = AVAILABLE_OPTIMIZERS[optimizer_type]
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        if self.learning_rate == 'auto':
            self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
        else:
            self.optimizer = self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.device = device
        self.module.to(device)

    def reset_parameters(self) -> None:
        """ Resets parameters of QLayers """
        for layer in self.module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()  # type: ignore

    def fit(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> "QModel":
        """ scikit-learn like fit method """
        x, y = self._x_y_to_tensor(x, y)
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        return self

    def _x_to_tensor(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(self.device)
        return x

    def _x_y_to_tensor(
        self,
        x: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame | pd.Series

    ) -> tuple[Tensor, Tensor]:
        x = self._x_to_tensor(x)
        if isinstance(y, np.ndarray):
            if y.dtype.kind == "i":
                y = torch.tensor(y, dtype=torch.int64)
            else:
                y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, pd.DataFrame):
            y = torch.tensor(y.values, dtype=torch.float32)
        elif isinstance(y, pd.Series):
            if y.dtype == np.dtype('int64'):
                y = torch.tensor(y.values, dtype=torch.int64)
            else:
                y = torch.tensor(y.values, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y tensors should have the same first dimension")
        y = y.to(self.device)
        return x, y

    def fit_predict(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> Tensor:
        """ scikit-learn like fit_predict method """
        self.fit(x, y)
        return self.predict(x)

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        pbar = tqdm(range(epochs), total=epochs, unit="epochs")
        for epoch in pbar:

            self.module.train()
            self._train_one_epoch(train_loader)

            self.module.eval()
            with torch.inference_mode():
                valid_loss = self._validate_one_epoch(validation_loader)
            pbar.set_postfix(loss=valid_loss, epoch=epoch + 1)

    def _train_one_epoch(self, train_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(train_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            outputs = self.module(x)
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
            outputs = self.module(x)
            loss = self.loss(outputs, y)
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses)

    def predict(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        """ scikit-learn like predict method """
        x = self._x_to_tensor(x)
        self.module.eval()
        with torch.inference_mode():
            result = self.module(x).cpu()
        return result

    def set_params(self, **params):
        """ scikit-learn like param setting method"""

        def _update_optimizer():

            if self.learning_rate == "auto":
                self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
            else:
                self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore

        if not params:
            return self
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {valid_params.keys()!r}."
                )
            if key == "device":
                self.device = value
                self.module.to(self.device)
            elif key == "optimizer_type":
                self.optimizer_type = value
                _update_optimizer()
            elif key == "learning_rate":
                self.learning_rate = value
                _update_optimizer()
            elif key == "module":
                self.module = value
                _update_optimizer()
        return self

    def to_torch_module(self) -> nn.Module:
        """Returns QModel's module with torch neural network.

        Returns:
            nn.Module: Torch neural network.
        """
        return self.module
