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

try:
    from ptseries.optimizers import HybridOptimizer
except ImportError:
    class HybridOptimizer():
        """Dummy HO"""
        # pylint: disable=too-few-public-methods

        def __init__(
            self,
            model,
            lr_classical=0.01,
            lr_quantum=0.01,
            optimizer_quantum='SGD',
            optimizer_classical='Adam',
            betas=(0.9, 0.999),
            spsa_resamplings=1,
            spsa_gamma_decay=0.101,
            spsa_alpha_decay=0.602
        ):
            pass

AVAILABLE_OPTIMIZERS: dict[str, type[Optimizer] | type[HybridOptimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad,
    optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS, HybridOptimizer]}


class QModel(BaseEstimator):
    """ Quantum model class

    Parameters
    ----------
    module: nn.Module
        pytorch Module representing the quantum or classical neural network.
    loss: Callable
        pytorch loss function to be used during training.
    optimizer_type: type[Optimizer] | str, default = "adamw"
        pytorch Optimizer class to be used during training.
    learning_rate: float | Literal['auto'], default = "auto"
        learning rate used by the optimizer, "auto" sets it to optimizer's default one.
    quantum_learning_rate: float | Literal['auto'], default = 'auto'
        learning rate for quantum layers used by the HybridOptimizer, "auto" sets it to optimizer's default one.
    batch_size: int, default = 1
        number of training examples in batch.
    epochs: int, default = 1
        number of epochs to train the model.
    validation_fraction: float, default = 0.2
       share of the training dataset to be used for validation.
    shuffle: bool, default = True
        whether to shuffle data every epoch.
    device: {"cpu","cuda","mps"}, default="cpu"
        the device neural network will be trained on.

    Attributes
    ----------
    optimizer: Optimizer
        pytorch optimizer object used during training
    loss_history: dict[str,list]
        history of loss values from the last fit call. dict contains keys 'training' and 'validation'
    """

    # pylint: disable=too-many-instance-attributes
    # Reasonable amount for model training

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
    device: Literal["cpu", "cuda", "mps"]
    metric: Literal["accuracy", "mse"] | None

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
        metric: Literal["accuracy", "mse"] | None = None
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
        self.module.to(device)

        self.loss_history = {
            'training': [],
            'validation': []
        }

    def reset_parameters(self) -> None:
        """ Resets parameters of layers """
        for layer in self.module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()  # type: ignore

    def fit(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> "QModel":
        """ scikit-learn like fit method
        trains the neural network based on training set (x,y).

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
            The training input samples of shape (n_samples, n_features).
        y: Tensor | np.array | pd.DataFrame | pd.Series
            The training target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self: QModel
            trained NN model
        """
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
        """ scikit-learn like fit_predict method
        trains the neural network based on training set (x,y) and predicts values for training examples x
        combines fit and predict methods into one.

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
            The training input samples of shape (n_samples, n_features).
        y: Tensor | np.array | pd.DataFrame | pd.Series
            The training target values of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        y_pred: Tensor
            The predicted values for the training examples x.
        """
        self.fit(x, y)
        return self.predict(x)

    @staticmethod
    def _accuracy(y_pred, y_gt):
        if y_gt.dim() == 2:
            y_gt = torch.argmax(y_gt, dim=1)
        if y_pred.dim() == 2:
            y_pred = torch.argmax(y_pred, dim=1)
        else:
            y_pred = torch.where(y_pred > 0.5, 1.0, 0.0)
        return (y_pred.eq(y_gt)).sum().item() / len(y_gt)

    @staticmethod
    def _mse(y_pred, y_gt):
        return ((y_pred - y_gt)**2).sum().item() / len(y_gt)

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        self.loss_history = {
            'training': [],
            'validation': []
        }
        pbar = tqdm(range(epochs), total=epochs, unit="epochs")
        for epoch in pbar:

            self.module.train()
            self._train_one_epoch(train_loader)

            self.module.eval()
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
            outputs = self.module(x)
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

        self.loss_history['training'].append(np.mean(losses))
        return np.mean(losses), np.mean(metrics)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> tuple[np.floating, np.floating]:
        losses = []
        metrics = []
        pbar = tqdm(validation_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            outputs = self.module(x)
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

        self.loss_history['validation'].append(np.mean(losses))
        return np.mean(losses), np.mean(metrics)

    def predict(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        """ scikit-learn like predict method
        predicts values for examples input examples x.

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
           The input samples of shape (n_samples, n_features).

        Returns
        -------
        y_pred: Tensor | np.ndarray | pd.DataFrame
            The predicted values for examples x.

        """
        x = self._x_to_tensor(x)
        self.module.eval()
        with torch.inference_mode():
            result = self.module(x).cpu()
        return result

    def set_params(self, **params):
        """ scikit-learn like param setting method
        allows changing parameters of the model set in constructor.

        Parameters
        ----------
        **params: dict
            Keyword arguments representing the parameters to be set.
        """

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
                self.module.to(self.device)
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

    def to_torch_module(self) -> nn.Module:
        """Returns QModel's module with torch neural network.

        Returns:
            nn.Module: Torch neural network.
        """
        return self.module
