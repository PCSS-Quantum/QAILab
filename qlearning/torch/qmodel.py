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
from ptseries.optimizers import HybridOptimizer

AVAILABLE_OPTIMIZERS: dict[str, type[Optimizer] | type[HybridOptimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad,
    optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS, HybridOptimizer]}


class QModel(nn.Module, BaseEstimator):
    """ Quantum model class """
    module: nn.Module
    loss: Callable
    optimizer_type: type[Optimizer] | type[HybridOptimizer]
    optimizer: Optimizer
    learning_rate: float | Literal['auto']
    quantum_learning_rate: float | Literal['auto']
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    def __init__(
        self,
        module: nn.Module,
        loss: Callable,
        optimizer_type: type[Optimizer] | type[HybridOptimizer] | str = 'adamw',
        learning_rate: float | Literal['auto'] = 'auto',
        quantum_learning_rate: float | Literal['auto'] = 'auto',
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        metric: Literal["accuracy", "mse"] = "accuracy"
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
        self.quantum_learning_rate = quantum_learning_rate
        if self.optimizer_type == HybridOptimizer:
            if self.quantum_learning_rate != 'auto' and self.learning_rate != "auto":
                self.optimizer = self.optimizer_type(self.module, lr_classical=self.learning_rate,
                                                     lr_quantum=self.quantum_learning_rate)  # type: ignore
            elif self.quantum_learning_rate != 'auto' and self.learning_rate == "auto":
                self.optimizer = self.optimizer_type(self.module, lr_quantum=self.quantum_learning_rate)  # type: ignore
            elif self.quantum_learning_rate == 'auto' and self.learning_rate != "auto":
                self.optimizer = self.optimizer_type(self.module, lr_classical=self.learning_rate)  # type: ignore
            else:
                self.optimizer = self.optimizer_type(self.module)  # type: ignore
        elif self.learning_rate == 'auto':
            self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
        else:
            self.optimizer = self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.device = device
        self.metric = metric
        self.to(device)

    def reset_parameters(self) -> None:
        """ Resets parameters of QLayers """
        for layer in self.module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()  # type: ignore

    def forward(self, input_tensor: Tensor) -> Tensor:
        """ Forward """
        return self.module(input_tensor)

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

    @staticmethod
    def _accuracy(y_pred, y_gt):
        return (torch.argmax(y_pred, dim=1).eq(y_gt)).sum().item() / len(y_gt)

    @staticmethod
    def _mse(y_pred, y_gt):
        return ((y_pred - y_gt)**2).sum().item() / len(y_gt)

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        pbar = tqdm(range(epochs), total=epochs, unit="epochs")
        for epoch in pbar:

            self.train()
            self._train_one_epoch(train_loader)

            self.eval()
            with torch.inference_mode():
                valid_loss, valid_metric = self._validate_one_epoch(validation_loader)
            if self.metric == "mse":
                pbar.set_postfix(loss=valid_loss, mse=valid_metric, epoch=epoch + 1)
            elif self.metric == "accuracy":
                pbar.set_postfix(loss=valid_loss, acc=valid_metric, epoch=epoch + 1)
            else:
                pbar.set_postfix(loss=valid_loss, epoch=epoch + 1)

    def _train_one_epoch(self, train_loader: DataLoader) -> tuple[np.floating, np.floating]:
        losses = []
        metrics = []
        pbar = tqdm(train_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            outputs = self(x)
            loss = self.loss(outputs, y)
            if self.metric == "mse":
                metrics.append(self._mse(outputs, y))
            elif self.metric == "accuracy":
                metrics.append(self._accuracy(outputs, y))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if self.metric == "mse":
                pbar.set_postfix(loss=loss.item(), mse=metrics[-1], batch=batch + 1)
            elif self.metric == "accuracy":
                pbar.set_postfix(loss=loss.item(), acc=metrics[-1], batch=batch + 1)
            else:
                pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses), np.mean(metrics)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> tuple[np.floating, np.floating]:
        losses = []
        metrics = []
        pbar = tqdm(validation_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            outputs = self(x)
            loss = self.loss(outputs, y)
            if self.metric == "mse":
                metrics.append(self._mse(outputs, y))
            elif self.metric == "accuracy":
                metrics.append(self._accuracy(outputs, y))
            losses.append(loss.item())
            if self.metric == "mse":
                pbar.set_postfix(loss=loss.item(), mse=metrics[-1], batch=batch + 1)
            elif self.metric == "accuracy":
                pbar.set_postfix(loss=loss.item(), acc=metrics[-1], batch=batch + 1)
            else:
                pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses), np.mean(metrics)

    def predict(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        """ scikit-learn like predict method """
        x = self._x_to_tensor(x)
        self.eval()
        with torch.inference_mode():
            result = self(x).cpu()
        return result

    def set_params(self, **params):
        """ scikit-learn like param setting method"""

        def _update_optimizer():

            if self.optimizer_type == HybridOptimizer:
                if self.quantum_learning_rate != 'auto' and self.learning_rate != "auto":
                    self.optimizer = self.optimizer_type(self.module, lr_classical=self.learning_rate,
                                                         lr_quantum=self.quantum_learning_rate)  # type: ignore
                elif self.quantum_learning_rate != 'auto' and self.learning_rate == "auto":
                    self.optimizer = self.optimizer_type(self.module, lr_quantum=self.quantum_learning_rate)  # type: ignore
                elif self.quantum_learning_rate == 'auto' and self.learning_rate != "auto":
                    self.optimizer = self.optimizer_type(self.module, lr_classical=self.learning_rate)  # type: ignore
                else:
                    self.optimizer = self.optimizer_type(self.module)  # type: ignore
            elif self.learning_rate == 'auto':
                self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
            else:
                self.optimizer = self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore

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
                self.to(self.device)
            elif key == "optimizer_type":
                self.optimizer_type = value
                _update_optimizer()
            elif key == "learning_rate":
                self.learning_rate = value
                _update_optimizer()
            elif key == "quantum_learning_rate":
                self.quantum_learning_rate = value
                _update_optimizer()
            elif key == "module":
                self.module = value
                _update_optimizer()
        return self
